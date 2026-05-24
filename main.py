import subprocess, sys, argparse, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

STEPS = {
    'preprocess': 'src/preprocess.py',
    'train': 'src/train.py',
    'optimize': 'src/optimize.py',
    'rank': 'src/rank.py',
    'check': 'src/check_exclusion.py'
}

def main():
    parser = argparse.ArgumentParser(description='GFP Optimizer Pipeline')
    parser.add_argument('--step', choices=list(STEPS.keys()) + ['all'], default='all')
    args = parser.parse_args()
    targets = list(STEPS.values()) if args.step == 'all' else [STEPS[args.step]]
    for script in targets:
        print(f'\n{"="*20} 🚀 Running: {script} {"="*20}')
        subprocess.run([sys.executable, str(PROJECT_ROOT / script)], check=True)
    print('\n✅ Pipeline completed.')

if __name__ == '__main__':
    main()
