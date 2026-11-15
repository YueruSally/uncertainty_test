"""Isolated multi-configuration supervision; historical modules are read-only."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPANDA = HERE.parent
LEGACY = EXPANDA / 'component_screen_b20_v1'
sys.path.insert(0, str(LEGACY))
from common import (ENGINE, read_json, write_json, digest, canonical_digest,
                    read_instance, divisor, sources as legacy_sources)
import learning as legacy_learning
import run as legacy_runner

FEATURE = 'multiscale-shared-v1-evaluation-over-50000'
COMPONENTS = ('I', 'C', 'M')


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen record changed; preserve this output: ' + str(path))
    else:
        write_json(path, value)


def verify(files):
    for path, expected in files.items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen file changed: ' + str(path))


def runtime_files():
    paths = legacy_sources() + list(HERE.glob('*.py')) + [HERE / 'protocol.json',
        EXPANDA / 'formal_icm_s0_v1/assets/data/data_expanded.xlsx',
        EXPANDA / 'formal_icm_s0_v1/assets/reward_normalization.json']
    return {str(p.resolve()): digest(p) for p in paths}


def progress_row(row, denominator=50000):
    """Reconstruct from saved evaluation count, never rescale an opaque token."""
    if int(row['evaluation']) != row['evaluation'] or row['evaluation'] < 0:
        raise ValueError('Invalid recorded evaluation count')
    value = dict(row)
    value['progress'] = row['evaluation'] / denominator
    value['feature_version'] = FEATURE
    return value
