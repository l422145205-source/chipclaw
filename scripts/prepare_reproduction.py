"""Create an isolated training copy without overwriting frozen results."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='reproduction')
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('Use a simple experiment directory name')
    target = ROOT / '.experiments' / args.name
    target.mkdir(parents=True, exist_ok=False)
    for name in ['chipclaw', 'scripts', 'web', 'data/ranking-v3-frozen', 'data/corpus-v3/index']:
        shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'chunks.jsonl'))
    for name in ['requirements-web.txt', 'requirements-v3.txt', 'requirements-v4.txt', 'data/corpus-v3/visual_subset.json', 'reports/v3-start/model_cache.json']:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    (target / 'reports').mkdir(exist_ok=True)
    print(target)


if __name__ == '__main__':
    main()
