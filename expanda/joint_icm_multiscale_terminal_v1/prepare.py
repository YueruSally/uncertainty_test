"""Freeze the existing training split and draw new paired development instances."""
from bridge import *
from dataclasses import asdict
import platform


def verify_shared(source):
    ready=read_json(source/'SHARED_READY.json');bundle=source/'shared_model/BUNDLE.json';b=read_json(bundle)
    if not ready['completed'] or not ready['full_budget_ready'] or ready['bundle_sha256']!=digest(bundle):raise ValueError('Shared supervision is not frozen/complete')
    if not b['coverage']['full_budget_ready'] or b['settings']['budget']!=50000:raise ValueError('Do not use historical B20000 pretraining')
    plan=read_json(source/'PLAN.json')
    verify(plan['files']);verify(read_json(source/'shared_model/PLAN.json')['files'])
    verify({str(source/'shared_model'/p):s for p,s in b['checkpoint_sha256'].items()})
    entries=read_json(source/'instances/catalog.json')['instances']
    if set(b['training_instances']['I'])&set(b['validation_instances']['I']):raise ValueError('Split overlap')
    for c in COMPONENTS:
        if b['training_instances'][c]!=b['training_instances']['I'] or b['validation_instances'][c]!=b['validation_instances']['I']:raise ValueError('Task splits differ')
    for cfg in b['settings']['configurations']:
        for role,count in [('training',6),('validation',2)]:
            group=[r for r in entries if r['configuration']==cfg and r['development_role']==role]
            if len(group)!=count:raise ValueError('Incomplete configuration split')
            expected=set(b['training_instances']['I'] if role=='training' else b['validation_instances']['I'])
            if not {r['instance_id'] for r in group}<=expected:raise ValueError('Bundle/catalog split mismatch')
    if {r['instance_id'] for r in entries}!=set(b['training_instances']['I'])|set(b['validation_instances']['I']):raise ValueError('Unexpected source instance')
    return b,entries


def generate_screen(root,source,settings,entries):
    import baseline_uncertainty as base
    from transformer_ppo.instance_catalog import _batches,_find_path_seed,SCENARIOS
    path=root/'screen_catalog.json'
    if path.exists():
        cat=read_json(path)
        if cat['generation_settings']!=settings:raise ValueError('Screen generation settings changed')
        for r in cat['instances']:
            if digest(root/'instances'/(r['instance_id']+'.json'))!=cat['files'][r['instance_id']]:raise ValueError('Screen instance changed')
        return cat['instances']
    source_catalog=read_json(Path(settings['source_catalog_default']))
    if source_catalog['catalog_digest']!=settings['source_catalog_digest'] or canonical_digest({k:v for k,v in source_catalog.items() if k!='catalog_digest'})!=source_catalog['catalog_digest']:
        raise ValueError('Wrong original OD split')
    od=[tuple(x) for x in source_catalog['train_od_pool']]
    if set(od)&{tuple(x) for x in source_catalog['test_od_pool']}:raise ValueError('OD split overlap')
    data=EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    if digest(data)!=source_catalog['source_data_sha256']:raise ValueError('Changed workbook')
    sample=entries[0];release=sample['release_time_definition']
    base.BORDER_EVENT_DEFINITIONS=base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    nodes,regions,_,_,_,arcs,timetables,raw,*_=base.load_network_from_extended(str(data))
    tt=base.build_timetable_dict(timetables);lookup=base.build_arc_lookup(arcs)
    known={r['batch_signature'] for r in entries}
    old_source=read_json(source/'PLAN.json')['source']
    known.update(r['batch_signature'] for r in read_json(Path(old_source)/'instances/catalog.json')['instances'])
    records=[]
    for index,cfg in enumerate(settings['configurations']):
        k,alpha=SCENARIOS[cfg];seed=settings['screen_instance_seed_base']+100*index
        batches=_batches(raw,od,k,alpha,seed,'train',release['maximum_h'],release['step_h'])
        signature=canonical_digest([asdict(b) for b in batches])
        if signature in known:raise ValueError('Duplicate instance; do not silently redraw')
        known.add(signature);name=f'joint-confirm-{cfg}-001'
        r={a:sample[a] for a in ('schema_version','alpha_definition','release_time_definition','quantity_definition','od_pool')}
        r.update(instance_id=name,configuration=cfg,K=k,deadline_window_alpha=alpha,split='train',development_role='screen',
            instance_seed=seed,path_seed=_find_path_seed(seed+1000000,nodes,regions,arcs,batches,tt,lookup),
            algorithm_seed=settings['screen_seeds'][0],ccp100_seed=settings['screen_scenario_seed_base']+index,oos5000_seed=None,
            source_data_sha256=digest(data),source_catalog_digest=source_catalog['catalog_digest'],batch_signature=signature,batches=[asdict(b) for b in batches])
        r['instance_digest']=canonical_digest(r);freeze(root/'instances'/(name+'.json'),r);read_instance(root/'instances'/(name+'.json'))
        records.append(r);print('[MULTI-PREPARE] '+name+' zero objective evaluations',flush=True)
    freeze(path,{'generation_settings':settings,'instances':records,
                 'files':{r['instance_id']:digest(root/'instances'/(r['instance_id']+'.json')) for r in records}})
    return records


def prepare(root,source,protocol):
    b,entries=verify_shared(source)
    settings={**b['settings'],**protocol}
    for r in entries:freeze(root/'instances'/(r['instance_id']+'.json'),r)
    screen=generate_screen(root,source,settings,entries)
    fileset={**files(),str(source/'shared_model/BUNDLE.json'):digest(source/'shared_model/BUNDLE.json'),
             str(root/'screen_catalog.json'):digest(root/'screen_catalog.json')}
    fileset.update({str(source/'shared_model'/p):s for p,s in b['checkpoint_sha256'].items()})
    fileset.update({str(root/'instances'/(r['instance_id']+'.json')):digest(root/'instances'/(r['instance_id']+'.json')) for r in entries+screen})
    plan={'settings':settings,'source':str(source),'files':fileset,'entries':entries,'screen':screen,
          'normalization':b['normalization'],'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':str(np.__version__)}}
    plan['sha256']=canonical_digest(plan);freeze(root/'PLAN.json',plan);return plan


def training_schedule(plan):
    s=plan['settings'];cfgs=s['configurations'];rows=[]
    by={c:sorted([r for r in plan['entries'] if r['development_role']=='training' and r['configuration']==c],key=lambda x:x['instance_id']) for c in cfgs}
    for i in range(s['ppo']['updates']*s['ppo']['episodes_per_update']):
        c=cfgs[i%len(cfgs)];r=by[c][(i//len(cfgs))%len(by[c])]
        rows.append({'update':i//s['ppo']['episodes_per_update']+1,'instance':r['instance_id'],'configuration':c,'seed':s['ppo']['training_seed_base']+i})
    return rows
