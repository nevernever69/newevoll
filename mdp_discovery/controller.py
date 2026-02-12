"""Evolution controller — the main loop for MDP Interface Discovery.

Ties together: LLM generation → crash filter → cascade training → MAP-Elites database.

Usage:
    config = Config.from_yaml("configs/default.yaml")
    controller = EvolutionController(config, task_description="Navigate to the goal.")
    controller.run()
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from mdp_discovery.config import Config
from mdp_discovery.database import ProgramDatabase
from mdp_discovery.evaluator import CandidateResult, CascadeEvaluator, EvalStage
from mdp_discovery.llm_client import LLMClient, LLMResponse
from mdp_discovery.prompts import PromptBuilder

logger = logging.getLogger(__name__)


class EvolutionController:
    """Main evolution loop for MDP Interface Discovery.

    Each iteration:
    1. Sample parent from database (or use from-scratch on first iteration)
    2. Build prompt with parent code, feedback, top/diverse programs
    3. Call LLM to generate candidate code
    4. Evaluate candidate (crash filter → short train → full train)
    5. Store result in database
    6. Handle island rotation and migration
    """

    def __init__(
        self,
        config: Config,
        task_description: str,
        library_context: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.config = config
        self.task_description = task_description
        self.checkpoint_dir = checkpoint_dir

        # Load library context
        if library_context is None:
            ctx_path = Path(__file__).parent.parent / "contexts" / "xminigrid_context.md"
            library_context = ctx_path.read_text()

        # Initialize components
        self.db = ProgramDatabase(config.database)
        self.evaluator = CascadeEvaluator(config)
        self.llm = LLMClient(config.llm)
        self.prompt_builder = PromptBuilder(
            config=config.prompt,
            library_context=library_context,
            task_description=task_description,
        )

        # Tracking
        self.iteration = 0
        self.total_llm_tokens = 0
        self.total_eval_time = 0.0
        self.start_time = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_iterations: Optional[int] = None) -> None:
        """Run the evolution loop.

        Args:
            max_iterations: Override config.max_iterations.
        """
        max_iter = max_iterations or self.config.max_iterations
        n = self.config.candidates_per_iteration
        self.start_time = time.time()
        self.iteration = 0

        logger.info(
            "Starting evolution: %d iterations, %d candidates/batch, env=%s, model=%s",
            max_iter,
            n,
            self.config.environment.env_id,
            self.config.llm.model_name,
        )

        executor = ThreadPoolExecutor(max_workers=n)
        try:
            while self.iteration < max_iter:
                batch_size = min(n, max_iter - self.iteration)
                try:
                    self._run_batch(executor, batch_size)
                except KeyboardInterrupt:
                    logger.info("Interrupted at iteration %d", self.iteration)
                    break
                except Exception:
                    logger.exception("Error in batch at iteration %d", self.iteration)
                    continue

                # Checkpoint
                if (
                    self.checkpoint_dir
                    and self.iteration % self.config.checkpoint_interval == 0
                ):
                    self._checkpoint()
        finally:
            executor.shutdown(wait=False)

        # Final checkpoint
        if self.checkpoint_dir:
            self._checkpoint()

        self._log_summary()

    # ------------------------------------------------------------------
    # Batch iteration
    # ------------------------------------------------------------------

    def _prepare_candidate(self):
        """Prepare prompt and parent info for one candidate (main thread).

        Returns:
            (prompt, parent_code, parent_id, generation)
        """
        parent = self.db.sample_parent() if self.db.programs else None

        if parent is not None:
            best = self.db.get_best_program()
            prompt = self.prompt_builder.build_prompt(
                parent_code=parent.code,
                parent_metrics=parent.metrics,
                top_programs=self.db.get_top_programs(
                    n=self.config.prompt.num_top_programs
                ),
                diverse_programs=self.db.get_diverse_programs(
                    n=self.config.prompt.num_diverse_programs
                ),
                failed_programs=self.db.get_recent_failures(
                    n=self.config.prompt.num_failure_examples
                ),
                best_metrics=best.metrics if best else None,
            )
            return prompt, parent.code, parent.id, parent.generation + 1
        else:
            prompt = self.prompt_builder.build_prompt(
                failed_programs=self.db.get_recent_failures(
                    n=self.config.prompt.num_failure_examples
                ),
            )
            return prompt, None, None, 0

    def _run_single_candidate(
        self, prompt, parent_code: Optional[str]
    ) -> Tuple[LLMResponse, Optional[CandidateResult]]:
        """Worker: LLM call + evaluate. Runs in thread pool."""
        response = self.llm.generate(prompt)

        if response.code is None:
            return response, None

        result = self.evaluator.evaluate(response.code)

        # Crash retry — give the LLM one chance to fix it
        if (
            result.stage == EvalStage.CRASHED
            and self.config.crash_filter.retry_on_crash
        ):
            retry_prompt = self.prompt_builder.build_prompt(
                parent_code=result.code,
                parent_metrics=None,
                failed_programs=[
                    {"code": result.code, "crash_result": result.crash_result}
                ],
            )
            retry_response = self.llm.generate(retry_prompt)
            # Accumulate retry tokens into the original response for bookkeeping
            response = LLMResponse(
                text=response.text,
                code=response.code,
                input_tokens=response.input_tokens + retry_response.input_tokens,
                output_tokens=response.output_tokens + retry_response.output_tokens,
                model=response.model,
            )

            if retry_response.code is not None:
                retry_result = self.evaluator.evaluate(retry_response.code)
                if retry_result.stage != EvalStage.CRASHED:
                    result = retry_result
                    logger.info(
                        "Retry succeeded: %s return=%.4f",
                        retry_result.stage.value,
                        retry_result.fitness,
                    )

        return response, result

    def _run_batch(self, executor: ThreadPoolExecutor, batch_size: int) -> None:
        """Prepare, submit, and collect one batch of candidates."""
        batch_start = time.time()

        # 1. Prepare batch on main thread (reads DB)
        batch = []
        for _ in range(batch_size):
            prompt, parent_code, parent_id, generation = self._prepare_candidate()
            batch.append((prompt, parent_code, parent_id, generation))

        logger.info(
            "[Iter %d] Submitting batch of %d candidate(s)...",
            self.iteration + 1,
            batch_size,
        )

        # 2. Submit all workers
        futures = {}
        for prompt, parent_code, parent_id, generation in batch:
            f = executor.submit(self._run_single_candidate, prompt, parent_code)
            futures[f] = (parent_id, generation)

        # 3. Collect results, store in DB (main thread, sequential)
        for f in as_completed(futures):
            parent_id, generation = futures[f]
            self.iteration += 1
            iter_time = time.time() - batch_start

            try:
                response, result = f.result()
            except Exception:
                logger.exception("[Iter %d] Worker failed", self.iteration)
                continue

            self.total_llm_tokens += response.input_tokens + response.output_tokens

            if result is None:
                logger.warning(
                    "[Iter %d] LLM returned no code fence, skipping",
                    self.iteration,
                )
                continue

            self.total_eval_time += result.eval_time

            # Store in database
            program = self.db.add(
                result,
                parent_id=parent_id,
                generation=generation,
            )

            # Log result
            self._log_iteration(result, program, response, iter_time)

            # Island management
            self.db.increment_generation()

            if self.iteration % self.config.database.num_islands == 0:
                self.db.next_island()

            if self.db.should_migrate():
                n_migrated = self.db.migrate()
                logger.info("[Iter %d] Migrated %d programs", self.iteration, n_migrated)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_iteration(
        self,
        result: CandidateResult,
        program,
        response,
        iter_time: float,
    ) -> None:
        """Log the outcome of a single iteration."""
        best = self.db.get_best_program()
        best_fitness = best.fitness if best else 0.0

        if result.stage == EvalStage.CRASHED:
            logger.info(
                "[Iter %d] CRASHED (stage %s): %s | best=%.4f | %.1fs",
                self.iteration,
                result.crash_result.stage_failed,
                result.crash_result.error_message[:80],
                best_fitness,
                iter_time,
            )
        elif result.stage == EvalStage.SHORT_TRAIN_REJECTED:
            logger.info(
                "[Iter %d] REJECTED (success=%.0f%%) | best=%.0f%% | %.1fs",
                self.iteration,
                result.fitness * 100,
                best_fitness * 100,
                iter_time,
            )
        else:
            is_new_best = program and best and program.id == best.id
            marker = " *** NEW BEST ***" if is_new_best else ""
            logger.info(
                "[Iter %d] %s success=%.0f%% length=%.1f | best=%.0f%% | %.1fs%s",
                self.iteration,
                result.stage.value,
                result.fitness * 100,
                result.metrics.get("final_length", 0) if result.metrics else 0,
                best_fitness * 100,
                iter_time,
                marker,
            )

    def _log_summary(self) -> None:
        """Log a final summary of the evolution run."""
        elapsed = time.time() - self.start_time
        stats = self.db.get_stats()
        best = self.db.get_best_program()

        logger.info("=" * 60)
        logger.info("Evolution complete")
        logger.info("  Iterations: %d", self.iteration)
        logger.info("  Wall time: %.1fs", elapsed)
        logger.info("  Total programs: %d", stats["total_programs"])
        logger.info("  Total eval time: %.1fs", self.total_eval_time)
        logger.info("  Total LLM tokens: %d", self.total_llm_tokens)
        logger.info("  Recent failures: %d", stats["recent_failures"])
        logger.info("  Grid cells filled: %s", stats["grid_cells_filled"])
        logger.info("  Island sizes: %s", stats["island_sizes"])
        if best:
            logger.info("  Best success rate: %.0f%%", best.fitness * 100)
            logger.info("  Best obs_dim: %s", best.obs_dim)
            logger.info("  Best program:\n%s", best.code)
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Save database, controller state, and best program to disk."""
        if not self.checkpoint_dir:
            return

        import json

        path = Path(self.checkpoint_dir)
        self.db.save(str(path / "database"))

        # Save controller metadata
        meta = {
            "iteration": self.iteration,
            "total_llm_tokens": self.total_llm_tokens,
            "total_eval_time": self.total_eval_time,
            "task_description": self.task_description,
            "config": self.config.to_dict(),
        }
        with open(path / "controller_state.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save best program as a standalone .py file
        self._save_best_interface()

        logger.info(
            "[Iter %d] Checkpoint saved to %s", self.iteration, self.checkpoint_dir
        )

    def _save_best_interface(self) -> None:
        """Write the best program to best_interface.py in the run directory."""
        if not self.checkpoint_dir:
            return

        best = self.db.get_best_program()
        if best is None:
            return

        path = Path(self.checkpoint_dir) / "best_interface.py"
        header = (
            f'"""\n'
            f'Best MDP Interface — evolved by MDP Interface Discovery\n'
            f'\n'
            f'Task:         {self.task_description}\n'
            f'Success rate: {best.fitness:.0%}\n'
            f'Obs dim:      {best.obs_dim}\n'
            f'Generation:   {best.generation}\n'
            f'Iteration:    {self.iteration}\n'
            f'"""\n\n'
        )
        path.write_text(header + best.code + "\n")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Resume from a checkpoint."""
        import json

        path = Path(checkpoint_dir)
        self.db.load(str(path / "database"))

        meta_path = path / "controller_state.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.iteration = meta.get("iteration", 0)
            self.total_llm_tokens = meta.get("total_llm_tokens", 0)
            self.total_eval_time = meta.get("total_eval_time", 0.0)

        logger.info(
            "Resumed from checkpoint: iteration=%d, %d programs",
            self.iteration,
            len(self.db.programs),
        )
