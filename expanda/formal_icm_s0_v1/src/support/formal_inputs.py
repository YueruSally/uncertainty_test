"""Freeze fresh S0 selection instances without evaluating any objectives."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil

import baseline_uncertainty as base
from runtime_common import digest, read_json, write_json
from transformer_ppo.instance_catalog import _batches, _find_path_seed, canonical_digest, read_instance


def validate_config(c):
    from experiment import METHODS
    if c['protocol'] != 'icm-formal-s0-component-screen-v1' or tuple(c['methods']) != METHODS:
        raise ValueError('Unsupported protocol or changed method list')
    if c['configuration'] != 'S0' or c['instance_count'] != 10 or len(c['algorithm_seeds']) != 30:
        raise ValueError('This release is fixed to 10 fresh S0 instances and 30 seeds')
    if len(set(c['algorithm_seeds'])) != 30 or c['ppo_enabled'] or c['final_test_opened']:
        raise ValueError('Invalid seeds or training/final-test mode')
    if (c['population_size'], c['maximum_generations'], c['evaluation_budget']) != (250, 200, 50000):
        raise ValueError('This formal release requires pop250 / max200 generations / B50000')
    if (c['scenario_count'], c['ccp_confidence'], c['candidate_pool'], c['oos_scenario_count']) != (100, .9, 8, 5000):
        raise ValueError('Scenario count, confidence, K8 or OOS contract changed')
    if c['workers'] not in (1, 2, 3, 4) or c['per_run_timeout_seconds'] < 60 or c['invocation_limit_seconds'] < 60:
        raise ValueError('Invalid resource limits')


def verify_catalog(root, config, data):
    cat = read_json(root / 'catalog.json')
    identity = dict(cat); claimed = identity.pop('catalog_digest')
    if canonical_digest(identity) != claimed or cat['settings_digest'] != canonical_digest(config):
        raise ValueError('Prepared catalog/config changed; use a new package/output, do not overwrite')
    if cat['count'] != 10 or cat['source_data_sha256'] != digest(data) or cat['final_test_instances_used']:
        raise ValueError('Catalog contract mismatch')
    if [x['instance_id'] for x in cat['instances']] != [f'formal-screen-S0-{i:03d}' for i in range(1,11)]:
        raise ValueError('Unexpected instance names')
    signatures = set()
    for item in cat['instances']:
        rec, _ = read_instance(root / (item['instance_id'] + '.json'))
        if rec['instance_digest'] != item['instance_digest'] or rec['batch_signature'] in signatures:
            raise ValueError('Changed/duplicate instance')
        if canonical_digest(rec['batches']) != rec['batch_signature'] or rec['configuration'] != 'S0':
            raise ValueError('Batch signature or configuration changed')
        for key in ('instance_seed','path_seed','ccp100_seed','oos5000_seed'):
            if rec[key] != item[key]: raise ValueError('Catalog/instance seed mismatch')
        signatures.add(rec['batch_signature'])
    return cat


def prepare_instances(package, config, source_catalog=None):
    assets = package / 'assets'; out = assets / 'formal_instances'
    data = assets / 'data/data_expanded.xlsx'
    if (out / 'catalog.json').exists():
        return verify_catalog(out, config, data)
    source = Path(source_catalog or config['source_catalog_default']).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError('需要原始实例目录中的 catalog.json（只读取 OD 划分，不读取测试结果）。缺少：' + str(source) + '\n请用 --source-catalog 指定服务器上的实际文件。')
    original = read_json(source); identity = dict(original); claimed = identity.pop('catalog_digest')
    if canonical_digest(identity) != claimed or claimed != config['expected_source_catalog_digest']:
        raise ValueError('Source catalog is not the previously frozen OD split')
    if original['source_data_sha256'] != digest(data):
        raise ValueError('Source workbook mismatch')
    train_od = [tuple(p) for p in original['train_od_pool']]
    test_od = {tuple(p) for p in original['test_od_pool']}
    if not train_od or set(train_od) & test_od:
        raise ValueError('Train/test OD pools overlap or are empty')
    known = set(); previous_seeds = set(); catalogs = [source, assets/'search_instances/catalog.json']
    prior_root = Path(config['prior_catalog_root_default'])
    if prior_root.is_dir(): catalogs.extend(sorted(prior_root.glob('*/catalog.json')))
    for path in catalogs:
        for item in read_json(path).get('instances', []):
            if not isinstance(item, dict): continue
            if item.get('batch_signature'): known.add(item['batch_signature'])
            for key in ('instance_seed','ccp100_seed','oos5000_seed'):
                if item.get(key) is not None: previous_seeds.add(item[key])
    sample = read_json(assets/'search_instances/prescreen-search-S0-001.json')
    out.mkdir(parents=True, exist_ok=True)
    generation_plan = {'settings_digest': canonical_digest(config), 'source_catalog_digest': claimed,
                       'source_data_sha256': digest(data), 'generator_sha256': digest(Path(__file__)),
                       'prior_catalogs_sha256': {str(p): digest(p) for p in catalogs}}
    plan_path = out/'GENERATION_PLAN.json'
    if plan_path.exists() and read_json(plan_path) != generation_plan:
        raise ValueError('Generation inputs changed; preserve existing files')
    write_json(plan_path, generation_plan)
    # Keep the exact source metadata in the package, without reading final-test outcomes.
    snapshot = assets/'source_catalog.json'
    if snapshot.exists() and digest(snapshot) != digest(source): raise ValueError('Source snapshot changed')
    if not snapshot.exists(): shutil.copy2(source, snapshot)
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    nodes, regions, _, _, _, arcs, timetables, raw, *_ = network
    tt = base.build_timetable_dict(timetables); lookup = base.build_arc_lookup(arcs)
    records = []
    for i in range(config['instance_count']):
        seed = config['instance_master_seed'] + i*100
        ccp = config['ccp_seed_base'] + i; oos = config['oos_seed_base'] + i
        if len({seed, ccp, oos}) != 3 or {seed, ccp, oos} & previous_seeds:
            raise ValueError('New seed namespace overlaps an existing catalog')
        batches = _batches(raw, train_od, 20, 1.0, seed, 'train',
                           sample['release_time_definition']['maximum_h'],
                           sample['release_time_definition']['step_h'])
        signature = canonical_digest([asdict(b) for b in batches])
        if signature in known: raise ValueError('Generated batches duplicate prior data; stop without seed retry')
        known.add(signature)
        name = f'formal-screen-S0-{i+1:03d}'
        path_seed = _find_path_seed(seed+config['path_seed_offset'], nodes, regions, arcs, batches, tt, lookup)
        record = {k: sample[k] for k in ('schema_version','configuration','K','deadline_window_alpha',
                                        'alpha_definition','release_time_definition','quantity_definition','od_pool')}
        record.update(instance_id=name, split='train', development_role='formal_component_screen',
                      instance_seed=seed, path_seed=path_seed, ccp100_seed=ccp, oos5000_seed=oos,
                      algorithm_seed=config['algorithm_seeds'][0], source_data_sha256=digest(data),
                      source_catalog_digest=claimed, batch_signature=signature,
                      batches=[asdict(b) for b in batches])
        record['instance_digest'] = canonical_digest(record)
        target = out/(name+'.json')
        if target.exists() and read_json(target) != record: raise ValueError('Existing instance changed: '+name)
        write_json(target, record)
        records.append({k: record[k] for k in ('instance_id','instance_seed','path_seed','ccp100_seed','oos5000_seed','instance_digest','batch_signature')})
        print(f'[FORMAL-PREPARE] {i+1}/10 {name}; zero objective evaluations', flush=True)
    cat = {**generation_plan, 'count': len(records), 'instances': records, 'configuration': 'S0',
           'purpose': 'fresh batch draws within frozen training OD domain; component selection, not final test',
           'training_instances_used': False, 'validation_instances_used': False,
           'final_test_instances_used': False, 'new_objective_evaluations': 0}
    cat['catalog_digest'] = canonical_digest(cat)
    write_json(out/'catalog.json', cat)
    return verify_catalog(out, config, data)
