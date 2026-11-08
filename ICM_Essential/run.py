#!/usr/bin/env python3
"""Check frozen inputs; --run invokes the unchanged common-seed experiment."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(assets):
    missing = []
    required = [assets / 'reward_normalization.json',
                assets / 'search_instances/catalog.json']
    required += [assets / f'search_instances/prescreen-search-S0-{i:03d}.json'
                 for i in range(1, 4)]
    required += [assets / c / 'BUNDLE.json' for c in ('I', 'C', 'M')]
    missing.extend(str(p) for p in required if not p.is_file())
    if missing:
        return missing, {}
    bundles = {}
    for component in ('I', 'C', 'M'):
        original = json.loads((assets / component / 'BUNDLE.json').read_text())
        relocated = copy.deepcopy(original)
        spec = relocated['models']['200'] if component == 'M' else relocated
        if len(spec['checkpoints']) != 3:
            raise ValueError(f'{component}: expected three frozen checkpoints')
        names, hashes = [], {}
        for old in spec['checkpoints']:
            path = assets / component / Path(old).name
            if not path.is_file():
                missing.append(str(path))
                continue
            expected = spec['checkpoint_sha256'][old]
            if digest(path) != expected:
                raise ValueError(f'Checkpoint hash mismatch: {path}')
            names.append(str(path.resolve()))
            hashes[str(path.resolve())] = expected
        spec['checkpoints'], spec['checkpoint_sha256'] = names, hashes
        bundles[component] = relocated
    if missing:
        return missing, {}
    prepared = ROOT / 'runtime_inputs'
    prepared.mkdir(exist_ok=True)
    paths = {}
    for component, bundle in bundles.items():
        path = prepared / f'{component}_BUNDLE.json'
        payload = json.dumps(bundle, sort_keys=True, indent=2) + '\n'
        if not path.exists() or path.read_text() != payload:
            path.write_text(payload)
        paths[component] = path
    return [], paths


def verify(bundles):
    sys.path.insert(0, str(ROOT / 'src/support'))
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import torch
    from initialization import FrozenIResidualGate
    from crossover import FrozenCResidualGate
    from mutation import FrozenScaleRanker
    from experiment import summarize
    torch.set_num_threads(1)
    policies = [FrozenIResidualGate(json.loads(bundles['I'].read_text()), digest),
                FrozenCResidualGate(json.loads(bundles['C'].read_text()), digest),
                FrozenScaleRanker(json.loads(bundles['M'].read_text()), digest)]
    for policy in policies:
        scaler = policy.scaler
        batch = torch.zeros(1, 8, 2, len(scaler['batch']['mean']))
        features = torch.zeros(1, 8, len(scaler['candidate']['mean']))
        context = torch.zeros(1, len(scaler['context']['mean']))
        marker = torch.zeros(1, 8, 1); marker[:, 0, 0] = 1
        with torch.inference_mode():
            for model in policy.models:
                values = model(batch, features, context, marker)
                assert values.shape == (1, 8) and torch.isfinite(values).all()
    stored = json.loads((ROOT / 'results/details.json').read_text())
    actual = summarize(stored['runs'])
    def equal(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: equal(a[k], b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b): equal(x, y)
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            assert np.isclose(a, b, rtol=1e-10, atol=1e-10)
        else: assert a == b
    for key in ('table', 'component_comparisons', 'combined_vs_single_components', 'paired_runs'):
        equal(actual[key], stored[key])
    print('通过：9 个模型加载和前向计算；240 次运行统计重算一致；没有新增搜索。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--out', type=Path, default=ROOT / 'new_runs')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument('--verify', action='store_true', help='Check models and recompute saved statistics without searches')
    parser.add_argument('--run', action='store_true',
                        help='Start 240 searches; without this flag only check files')
    args = parser.parse_args()
    assets = args.assets.resolve()
    missing, bundles = prepare(assets)
    if missing:
        print('缺少服务器冻结输入，尚不能复现本轮模型结果：')
        for item in missing:
            print('  ' + item)
        print('源码与已有汇总结果仍可直接阅读。详见 README_CN.txt。')
        return 2
    print('冻结输入和九个 checkpoint 哈希检查通过。')
    if args.verify:
        verify(bundles)
        return 0
    if not args.run:
        print('未启动搜索；加 --run 才会运行实验。')
        return 0
    source = ROOT / 'src'
    command = [sys.executable, '-u',
        str(source / 'experiment.py'),
        '--data', str(assets / 'data/data_expanded.xlsx'),
        '--search-instances', str(assets / 'search_instances'),
        '--normalization', str(assets / 'reward_normalization.json'),
        '--i-bundle', str(bundles['I']), '--c-bundle', str(bundles['C']),
        '--m-bundle', str(bundles['M']), '--out', str(args.out.resolve()),
        '--workers', str(args.workers)]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'}
    return subprocess.call(command, cwd=source, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
