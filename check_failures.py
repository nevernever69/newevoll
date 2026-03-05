import json

with open('runs/Go1PushRecovery_20260305_105057/controller_state.json') as f:
    data = json.load(f)

if 'recent_failures' in data:
    print('Recent failures:')
    for i, fail in enumerate(data.get('recent_failures', [])[:3], 1):
        print(f'\n{"="*60}')
        print(f'Failure {i}')
        print("="*60)
        if 'code' in fail:
            print('\nGenerated Code:')
            print(fail['code'])
        if 'error' in fail:
            print('\nError:', fail.get('error', 'No error message'))
        if 'stage' in fail:
            print('Stage:', fail.get('stage'))
else:
    print('No recent_failures in controller state')
    print('Keys:', list(data.keys()))
