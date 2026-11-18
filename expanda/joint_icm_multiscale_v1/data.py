"""Audit old feedback, exclude screening data, and assemble signed HV labels."""
from bootstrap import *
from collections import Counter
import numpy as np


def audit_source(source, settings):
    plan = read_json(source / 'PLAN.json')
    body = {k: v for k, v in plan.items() if k != 'sha256'}
    if canonical_digest(body) != plan['sha256']:
        raise ValueError('Source plan digest invalid')
    protocol = plan['protocol']
    if (protocol['protocol'] != 'component-screen-b20-v1' or
            protocol['population'] != 250 or protocol['budget'] != 20000 or
            protocol['scenario_count'] != 100 or protocol['confidence'] != .9):
        raise ValueError('Expected completed S0-S8 B20000 / pop250 / CCP100 collection')
    verify({str(EXPANDA / p): sha for p, sha in plan['files'].items()})
    catalog = read_json(source / 'instances/catalog.json')
    if (canonical_digest({k:v for k,v in catalog.items() if k!='catalog_digest'}) != catalog['catalog_digest']
            or catalog['protocol_digest'] != canonical_digest(protocol)):
        raise ValueError('Source catalog changed')
    entries = [e for e in catalog['instances'] if e['development_role'] in ('training','validation')]
    files = {str(source / 'PLAN.json'): digest(source / 'PLAN.json'),
             str(source / 'instances/catalog.json'): digest(source / 'instances/catalog.json')}
    instances = []; identities = set(); signatures = set(); counts = Counter(); state_counts = Counter()
    folders = []
    for entry_index,entry in enumerate(entries):
        name = entry['instance_id']; role = entry['development_role']; cfg = entry['configuration']
        path = source / 'instances' / (name + '.json'); info, _ = read_instance(path)
        if (name in identities or info['batch_signature'] in signatures or
                info['instance_digest'] != entry['instance_digest'] or
                info['configuration'] != cfg or info['development_role'] != role):
            raise ValueError('Duplicate or inconsistent instance split')
        identities.add(name); signatures.add(info['batch_signature']); counts[(cfg, role)] += 1
        instances.append(info); files[str(path)] = digest(path)
        for arm, components in (('COLLECT_I','I'),('COLLECT_CM','CM')):
            folder = source / 'collection' / name / arm
            done = legacy_runner.completed(folder, plan['sha256'])
            if not done:
                raise ValueError('Source collection incomplete: ' + str(folder))
            result = read_json(folder / 'DATA.json')
            if result['instance'] != name or result['role'] != role or result['configuration'] != cfg:
                raise ValueError('Source feedback identity mismatch')
            if not protocol['population'] <= result['actual_search_evaluations'] <= protocol['budget']:
                raise ValueError('Invalid source search cost')
            for c in components:
                rows = [r for r in result['rows'] if r['component'] == c]
                validate_rows(rows, info, c, protocol['budget'])
                if not rows or not all(result['phase_counts'][c]):
                    raise ValueError('Missing source sampling phase: ' + name + '/' + c)
                state_counts[(c, role)] += len(rows)
            folders.append(folder)
            for p in (folder / 'DATA.json', folder / 'RUN_COMPLETE.json'):
                files[str(p)] = digest(p)
        if (entry_index+1)%8==0:
            print(f'[MULTI-AUDIT] verified {entry_index+1}/{len(entries)} instances',flush=True)
    expected = {(cfg, role): count for cfg in settings['configurations']
                for role, count in [('training',6),('validation',2)]}
    if dict(counts) != expected:
        raise ValueError('Need all nine configurations, 54 training / 18 validation instances')
    report = {'completed': True, 'source': str(source), 'training_instances':54,
              'validation_instances':18, 'training_states':{c:state_counts[c,'training'] for c in COMPONENTS},
              'validation_states':{c:state_counts[c,'validation'] for c in COMPONENTS},
              'source_budget':20000, 'target_budget':50000,
              'historical_I_rewrites_fixed_seeds':True,
              'reuse_role':'pretraining diagnostics only; no B50000 readiness claim',
              'full_collection_needed':'C/M missing evaluations 20000-50000; I must preserve fixed Base seeds',
              'screen_instances_read_for_training':False, 'new_objective_evaluations':0}
    return instances, folders, files, report


def validate_rows(rows, info, component, budget):
    for r in rows:
        if (r['component'] != component or r['instance'] != info['instance_id'] or
                r['configuration'] != info['configuration'] or len(r['legal_mask']) != 8 or
                not r['legal_mask'][0] or len(r['labels']) != 8 or
                not 0 <= r['evaluation'] <= budget):
            raise ValueError('Invalid candidate row identity/mask/evaluation')
        if len(r['candidate_batch_tokens']) != 8 or any(len(t) != info['K'] for t in r['candidate_batch_tokens']):
            raise ValueError('Batch-token count differs from configuration')
        pair_size = 2 if component == 'C' else 1
        if any(len(pair) != pair_size for pair in r['labels']):
            raise ValueError('C requires two child outcomes; I/M require one')
        for pair in r['labels']:
            for child in pair:
                if child['feasible'] and (child['objectives'] is None or
                        np.shape(child['objectives']) != (3,) or not np.isfinite(child['objectives']).all()):
                    raise ValueError('Invalid feasible objective vector')
        if not np.isfinite(r['candidate_features']).all() or not np.isfinite(r['candidate_batch_tokens']).all():
            raise ValueError('Nonfinite features')


def load_dataset(folders, denominator):
    by = {c:{'training':[], 'validation':[]} for c in COMPONENTS}
    training_folders = []
    for folder in folders:
        d = read_json(folder / 'DATA.json')
        if d['role'] not in ('training','validation'):
            raise ValueError('Screen/test labels forbidden')
        if d['role'] == 'training': training_folders.append(folder)
        for r in d['rows']:
            by[r['component']][d['role']].append(progress_row(r, denominator))
    for c in COMPONENTS:
        tr = {r['instance'] for r in by[c]['training']}
        va = {r['instance'] for r in by[c]['validation']}
        if not tr or not va or tr & va:
            raise ValueError('Missing or overlapping train/validation data')
        if c != 'I' and (tr != {r['instance'] for r in by['I']['training']} or
                        va != {r['instance'] for r in by['I']['validation']}):
            raise ValueError('Component splits differ')
    norm = legacy_learning.fit_normalization(training_folders)
    return by, norm
