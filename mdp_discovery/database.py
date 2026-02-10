"""MAP-Elites database with island model for MDP Interface Discovery.

Stores evaluated programs in a feature-dimension grid. Supports:
- MAP-Elites: programs placed in bins by (reward_complexity, obs_dimensionality)
- Island model: multiple subpopulations with periodic migration
- Sampling: parent selection via exploration/exploitation/weighted ratios
- Retrieval: top programs, diverse programs, recent failures for prompts

Adapted from OpenEvolve's ProgramDatabase but simplified for our use case.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from mdp_discovery.config import DatabaseConfig
from mdp_discovery.evaluator import CandidateResult, EvalStage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Program record
# ---------------------------------------------------------------------------


@dataclass
class Program:
    """A single evaluated MDP interface program."""

    id: str
    code: str
    metrics: Dict[str, float] = field(default_factory=dict)
    obs_dim: Optional[int] = None
    stage: str = ""  # EvalStage value
    generation: int = 0
    parent_id: Optional[str] = None
    island: int = 0
    timestamp: float = field(default_factory=time.time)

    # Feature coordinates in the MAP-Elites grid
    feature_coords: List[int] = field(default_factory=list)

    @property
    def fitness(self) -> float:
        return self.metrics.get("success_rate", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Program:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_candidate_result(
        cls,
        result: CandidateResult,
        parent_id: Optional[str] = None,
        generation: int = 0,
        island: int = 0,
    ) -> Program:
        """Create a Program from an evaluator CandidateResult."""
        return cls(
            id=str(uuid.uuid4()),
            code=result.code,
            metrics=result.metrics or {},
            obs_dim=result.obs_dim,
            stage=result.stage.value,
            generation=generation,
            parent_id=parent_id,
            island=island,
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Failed program record (lightweight, for prompt feedback)
# ---------------------------------------------------------------------------


@dataclass
class FailedProgram:
    """A program that crashed or scored very poorly."""

    code: str
    crash_result: Any = None  # CrashFilterResult or None
    metrics: Optional[Dict[str, float]] = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _count_reward_complexity(code: str) -> int:
    """Count AST nodes in the compute_reward function as a complexity proxy."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
            return sum(1 for _ in ast.walk(node))
    return 0


def compute_features(
    program: Program, feature_dimensions: List[str]
) -> Dict[str, float]:
    """Compute raw feature values for a program.

    Supported dimensions:
    - reward_complexity: AST node count in compute_reward
    - obs_dimensionality: observation vector size
    """
    features: Dict[str, float] = {}
    for dim in feature_dimensions:
        if dim == "reward_complexity":
            features[dim] = float(_count_reward_complexity(program.code))
        elif dim == "obs_dimensionality":
            features[dim] = float(program.obs_dim or 0)
        elif dim in program.metrics:
            features[dim] = program.metrics[dim]
        else:
            features[dim] = 0.0
    return features


# ---------------------------------------------------------------------------
# MAP-Elites Database
# ---------------------------------------------------------------------------


class ProgramDatabase:
    """MAP-Elites + island model database for evolutionary search.

    Usage:
        db = ProgramDatabase(config)

        # Add evaluated candidates
        program = db.add(candidate_result, parent_id=..., generation=...)

        # Sample parent for next iteration
        parent = db.sample_parent()

        # Get data for prompt builder
        top = db.get_top_programs(n=3)
        diverse = db.get_diverse_programs(n=2)
        failures = db.get_recent_failures(n=2)
        best = db.get_best_program()
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)

        # All programs by ID
        self.programs: Dict[str, Program] = {}

        # Island populations (sets of program IDs)
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]

        # MAP-Elites grids: per-island, feature_key → program_id
        self.grid: List[Dict[str, str]] = [{} for _ in range(config.num_islands)]

        # Archive of top programs (global)
        self.archive: List[str] = []

        # Recent failures for feedback
        self.recent_failures: List[FailedProgram] = []
        self._max_failures = 20  # ring buffer size

        # Tracking
        self.best_program_id: Optional[str] = None
        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_gen: int = 0
        self.total_added: int = 0

        # Feature scaling stats (min/max per dimension, for binning)
        self._feature_stats: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(
        self,
        result: CandidateResult,
        parent_id: Optional[str] = None,
        generation: int = 0,
    ) -> Optional[Program]:
        """Add an evaluated candidate to the database.

        Crashed candidates are stored as failures only (for prompt feedback).
        Trained candidates are placed in the MAP-Elites grid.

        Returns the Program if it was added to the grid, None otherwise.
        """
        # Record failures for feedback
        if not result.passed or result.stage == EvalStage.CRASHED:
            self.add_failure(
                FailedProgram(
                    code=result.code,
                    crash_result=result.crash_result,
                    metrics=result.metrics,
                )
            )
            return None

        # Low-fitness programs that trained but performed very poorly
        if result.fitness == 0.0:
            self.add_failure(
                FailedProgram(
                    code=result.code,
                    metrics=result.metrics,
                )
            )

        # Determine island
        island = self.current_island
        if parent_id and parent_id in self.programs:
            island = self.programs[parent_id].island

        # Create program record
        program = Program.from_candidate_result(
            result,
            parent_id=parent_id,
            generation=generation,
            island=island,
        )

        # Compute features and bin
        features = compute_features(program, self.config.feature_dimensions)
        coords = self._bin_features(features)
        program.feature_coords = coords
        feature_key = self._coords_to_key(coords)

        # MAP-Elites placement: keep if cell is empty or new program is better
        island_grid = self.grid[island]
        existing_id = island_grid.get(feature_key)

        should_place = existing_id is None
        if not should_place:
            existing = self.programs.get(existing_id)
            if existing is None or program.fitness > existing.fitness:
                should_place = True

        if should_place:
            # Remove old program from island if replacing
            if existing_id and existing_id in self.programs:
                self.islands[island].discard(existing_id)

            self.programs[program.id] = program
            self.islands[island].add(program.id)
            island_grid[feature_key] = program.id
            self._update_archive(program)
            self._update_best(program)
            self.total_added += 1

            logger.debug(
                "Added program %s to island %d cell %s (fitness=%.4f)",
                program.id[:8],
                island,
                feature_key,
                program.fitness,
            )
            return program

        # Program didn't beat the existing cell occupant
        # Still store it (it may be sampled as a non-grid program)
        self.programs[program.id] = program
        self.islands[island].add(program.id)
        self.total_added += 1
        return program

    def add_failure(self, failure: FailedProgram) -> None:
        """Record a failed program for prompt feedback."""
        self.recent_failures.append(failure)
        if len(self.recent_failures) > self._max_failures:
            self.recent_failures.pop(0)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_parent(self) -> Optional[Program]:
        """Sample a parent program from the current island.

        Uses exploration/exploitation/weighted ratios from config:
        - exploration_ratio: uniform random from island
        - exploitation_ratio: random from global archive (elites)
        - remainder: fitness-weighted from island
        """
        if not self.programs:
            return None

        r = self.rng.random()

        if r < self.config.exploration_ratio:
            return self._sample_exploration()
        elif r < self.config.exploration_ratio + self.config.exploitation_ratio:
            return self._sample_exploitation()
        else:
            return self._sample_weighted()

    def _sample_exploration(self) -> Optional[Program]:
        """Uniform random from current island."""
        island_ids = [
            pid for pid in self.islands[self.current_island]
            if pid in self.programs
        ]
        if not island_ids:
            return self._sample_any()
        return self.programs[self.rng.choice(island_ids)]

    def _sample_exploitation(self) -> Optional[Program]:
        """Random from the global archive (elite programs)."""
        valid = [pid for pid in self.archive if pid in self.programs]
        if not valid:
            return self._sample_any()
        return self.programs[self.rng.choice(valid)]

    def _sample_weighted(self) -> Optional[Program]:
        """Fitness-weighted selection from current island."""
        island_ids = [
            pid for pid in self.islands[self.current_island]
            if pid in self.programs
        ]
        if not island_ids:
            return self._sample_any()

        programs = [self.programs[pid] for pid in island_ids]
        fitnesses = [p.fitness for p in programs]

        # Shift so all weights are positive
        min_f = min(fitnesses)
        weights = [f - min_f + 1e-6 for f in fitnesses]

        return self.rng.choices(programs, weights=weights, k=1)[0]

    def _sample_any(self) -> Optional[Program]:
        """Fallback: sample from any program in the database."""
        if not self.programs:
            return None
        return self.programs[self.rng.choice(list(self.programs.keys()))]

    # ------------------------------------------------------------------
    # Retrieval (for PromptBuilder)
    # ------------------------------------------------------------------

    def get_best_program(self) -> Optional[Program]:
        """Get the globally best program by fitness."""
        if self.best_program_id and self.best_program_id in self.programs:
            return self.programs[self.best_program_id]
        return None

    def get_top_programs(
        self, n: int = 3, island: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get top N programs by fitness, formatted for PromptBuilder.

        Returns list of {"code": str, "metrics": dict}.
        """
        if island is not None:
            candidates = [
                self.programs[pid]
                for pid in self.islands[island]
                if pid in self.programs
            ]
        else:
            candidates = list(self.programs.values())

        candidates.sort(key=lambda p: p.fitness, reverse=True)

        return [
            {"code": p.code, "metrics": p.metrics}
            for p in candidates[:n]
        ]

    def get_diverse_programs(self, n: int = 2) -> List[Dict[str, Any]]:
        """Get N programs from different MAP-Elites cells.

        Picks programs from the most distant feature coordinates to maximize
        diversity. Formatted for PromptBuilder.
        """
        if not self.programs:
            return []

        # Collect unique grid cells across all islands
        cell_programs: Dict[str, Program] = {}
        for island_idx, island_grid in enumerate(self.grid):
            for key, pid in island_grid.items():
                if pid in self.programs:
                    p = self.programs[pid]
                    if key not in cell_programs or p.fitness > cell_programs[key].fitness:
                        cell_programs[key] = p

        if not cell_programs:
            return []

        # Sort by feature key to spread across the grid
        items = sorted(cell_programs.items(), key=lambda x: x[0])

        # Pick evenly spaced programs across the sorted cells
        if len(items) <= n:
            selected = [p for _, p in items]
        else:
            step = len(items) / n
            selected = [items[int(i * step)][1] for i in range(n)]

        return [
            {"code": p.code, "metrics": p.metrics}
            for p in selected
        ]

    def get_recent_failures(self, n: int = 2) -> List[Dict[str, Any]]:
        """Get N most recent failures, formatted for PromptBuilder.

        Returns list of:
        - {"code": str, "crash_result": CrashFilterResult} for crashes
        - {"code": str, "metrics": dict} for low-fitness programs
        """
        failures = self.recent_failures[-n:]
        result = []
        for f in failures:
            entry: Dict[str, Any] = {"code": f.code}
            if f.crash_result is not None:
                entry["crash_result"] = f.crash_result
            if f.metrics is not None:
                entry["metrics"] = f.metrics
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Island management
    # ------------------------------------------------------------------

    def next_island(self) -> int:
        """Advance to the next island (round-robin)."""
        self.current_island = (self.current_island + 1) % self.config.num_islands
        return self.current_island

    def increment_generation(self, island: Optional[int] = None) -> None:
        """Increment the generation counter for an island."""
        idx = island if island is not None else self.current_island
        self.island_generations[idx] += 1

    def should_migrate(self) -> bool:
        """Check if it's time for inter-island migration."""
        max_gen = max(self.island_generations) if self.island_generations else 0
        return (max_gen - self.last_migration_gen) >= self.config.migration_interval

    def migrate(self) -> int:
        """Migrate top programs between islands (ring topology).

        Returns the number of programs migrated.
        """
        num_islands = self.config.num_islands
        if num_islands < 2:
            return 0

        migrated = 0

        for src_idx in range(num_islands):
            src_ids = [
                pid for pid in self.islands[src_idx] if pid in self.programs
            ]
            if not src_ids:
                continue

            # Select top programs to migrate
            src_programs = sorted(
                [self.programs[pid] for pid in src_ids],
                key=lambda p: p.fitness,
                reverse=True,
            )
            n_migrate = max(1, int(len(src_programs) * self.config.migration_rate))
            migrants = src_programs[:n_migrate]

            # Target: next island in ring
            dst_idx = (src_idx + 1) % num_islands

            for prog in migrants:
                # Create a copy for the destination island
                new_id = str(uuid.uuid4())
                migrant = Program(
                    id=new_id,
                    code=prog.code,
                    metrics=dict(prog.metrics),
                    obs_dim=prog.obs_dim,
                    stage=prog.stage,
                    generation=prog.generation,
                    parent_id=prog.id,
                    island=dst_idx,
                    timestamp=time.time(),
                    feature_coords=list(prog.feature_coords),
                )

                # Place in destination grid
                feature_key = self._coords_to_key(migrant.feature_coords)
                dst_grid = self.grid[dst_idx]
                existing_id = dst_grid.get(feature_key)

                should_place = existing_id is None
                if not should_place:
                    existing = self.programs.get(existing_id)
                    if existing is None or migrant.fitness > existing.fitness:
                        should_place = True

                if should_place:
                    if existing_id and existing_id in self.programs:
                        self.islands[dst_idx].discard(existing_id)
                    self.programs[new_id] = migrant
                    self.islands[dst_idx].add(new_id)
                    dst_grid[feature_key] = new_id
                    self._update_archive(migrant)
                    self._update_best(migrant)
                    migrated += 1

        self.last_migration_gen = max(self.island_generations)
        logger.info("Migration complete: %d programs migrated", migrated)
        return migrated

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for logging."""
        best = self.get_best_program()
        return {
            "total_programs": len(self.programs),
            "total_added": self.total_added,
            "archive_size": len(self.archive),
            "recent_failures": len(self.recent_failures),
            "best_fitness": best.fitness if best else None,
            "best_id": self.best_program_id,
            "current_island": self.current_island,
            "island_sizes": [
                len([pid for pid in isl if pid in self.programs])
                for isl in self.islands
            ],
            "island_generations": list(self.island_generations),
            "grid_cells_filled": [len(g) for g in self.grid],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save database state to disk."""
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "best_program_id": self.best_program_id,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_gen": self.last_migration_gen,
            "total_added": self.total_added,
            "archive": self.archive,
            "islands": [list(isl) for isl in self.islands],
            "grid": [dict(g) for g in self.grid],
            "feature_stats": self._feature_stats,
        }
        with open(base / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save programs
        programs_dir = base / "programs"
        programs_dir.mkdir(exist_ok=True)
        for pid, prog in self.programs.items():
            with open(programs_dir / f"{pid}.json", "w") as f:
                json.dump(prog.to_dict(), f, indent=2)

        logger.info("Database saved to %s (%d programs)", path, len(self.programs))

    def load(self, path: str) -> None:
        """Load database state from disk."""
        base = Path(path)

        with open(base / "metadata.json") as f:
            metadata = json.load(f)

        self.best_program_id = metadata.get("best_program_id")
        self.current_island = metadata.get("current_island", 0)
        self.island_generations = metadata.get("island_generations", [0] * self.config.num_islands)
        self.last_migration_gen = metadata.get("last_migration_gen", 0)
        self.total_added = metadata.get("total_added", 0)
        self.archive = metadata.get("archive", [])
        self.islands = [set(isl) for isl in metadata.get("islands", [])]
        self.grid = [dict(g) for g in metadata.get("grid", [])]
        self._feature_stats = metadata.get("feature_stats", {})

        # Pad if config changed num_islands
        while len(self.islands) < self.config.num_islands:
            self.islands.append(set())
            self.grid.append({})
            self.island_generations.append(0)

        # Load programs
        programs_dir = base / "programs"
        self.programs = {}
        if programs_dir.exists():
            for fpath in programs_dir.glob("*.json"):
                with open(fpath) as f:
                    d = json.load(f)
                prog = Program.from_dict(d)
                self.programs[prog.id] = prog

        logger.info("Database loaded from %s (%d programs)", path, len(self.programs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bin_features(self, features: Dict[str, float]) -> List[int]:
        """Bin raw feature values into grid coordinates."""
        coords = []
        for dim in self.config.feature_dimensions:
            value = features.get(dim, 0.0)
            self._update_feature_stats(dim, value)
            scaled = self._scale_feature(dim, value)
            bin_idx = int(scaled * self.config.feature_bins)
            bin_idx = max(0, min(self.config.feature_bins - 1, bin_idx))
            coords.append(bin_idx)
        return coords

    def _update_feature_stats(self, dim: str, value: float) -> None:
        """Track min/max for adaptive feature scaling."""
        if dim not in self._feature_stats:
            self._feature_stats[dim] = {"min": value, "max": value}
        else:
            stats = self._feature_stats[dim]
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)

    def _scale_feature(self, dim: str, value: float) -> float:
        """Scale a feature value to [0, 1] using min-max normalization."""
        stats = self._feature_stats.get(dim)
        if stats is None:
            return 0.5
        range_ = stats["max"] - stats["min"]
        if range_ < 1e-8:
            return 0.5
        return (value - stats["min"]) / range_

    @staticmethod
    def _coords_to_key(coords: List[int]) -> str:
        """Convert grid coordinates to a hashable key."""
        return "-".join(str(c) for c in coords)

    def _update_archive(self, program: Program) -> None:
        """Add program to the global archive if it qualifies."""
        if program.id in self.archive:
            return

        if len(self.archive) < self.config.archive_size:
            self.archive.append(program.id)
        else:
            # Replace the worst program in the archive
            worst_id = None
            worst_fitness = float("inf")
            for pid in self.archive:
                p = self.programs.get(pid)
                if p is None:
                    worst_id = pid
                    break
                if p.fitness < worst_fitness:
                    worst_fitness = p.fitness
                    worst_id = pid

            if worst_id and (
                self.programs.get(worst_id) is None
                or program.fitness > worst_fitness
            ):
                self.archive.remove(worst_id)
                self.archive.append(program.id)

    def _update_best(self, program: Program) -> None:
        """Update the globally best program if this one is better."""
        if self.best_program_id is None:
            self.best_program_id = program.id
            return
        best = self.programs.get(self.best_program_id)
        if best is None or program.fitness > best.fitness:
            self.best_program_id = program.id
