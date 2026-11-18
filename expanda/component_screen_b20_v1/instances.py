from common import *
from dataclasses import asdict
import numpy as np
import baseline_uncertainty as base
from transformer_ppo.instance_catalog import _batches,_find_path_seed

def prepare(root,settings,source):
    path=root/'instances/catalog.json'
    if path.exists():
        cat=read_json(path);v=dict(cat);claimed=v.pop('catalog_digest')
        if canonical_digest(v)!=claimed or cat['protocol_digest']!=canonical_digest(settings):raise ValueError('Changed instance plan')
        for r in cat['instances']:
            info,_=read_instance(root/'instances'/f'{r["instance_id"]}.json')
            if info['instance_digest']!=r['instance_digest']:raise ValueError('Instance hash changed')
        return cat
    src=read_json(source);v=dict(src);claimed=v.pop('catalog_digest')
    if claimed!=settings['source_catalog_digest'] or canonical_digest(v)!=claimed:raise ValueError('Original OD split digest differs')
    data=ROOT.parent/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    if digest(data)!=src['source_data_sha256']:raise ValueError('Network workbook changed')
    train=[tuple(x) for x in src['train_od_pool']]
    if set(train)&{tuple(x) for x in src['test_od_pool']}:raise ValueError('OD pools overlap')
    sample=read_json(ROOT.parent/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
    known={r.get('batch_signature') for r in src['instances'] if isinstance(r,dict)}
    base.BORDER_EVENT_DEFINITIONS=base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network=base.load_network_from_extended(str(data));nodes,regions,_,_,_,arcs,timetables,raw,*_=network
    tt=base.build_timetable_dict(timetables);lookup=base.build_arc_lookup(arcs)
    records=[];index=0
    for configuration in settings['configurations']:
        k,alpha=SCENARIOS[configuration]
        for role,count in [('training',settings['train_per_configuration']),('validation',settings['validation_per_configuration']),('screen',settings['screen_per_configuration'])]:
            for i in range(count):
                seed=settings['instance_master_seed']+index*100;name=f'b20-{role}-{configuration}-{i+1:03d}'
                batches=_batches(raw,train,k,alpha,seed,'train',sample['release_time_definition']['maximum_h'],sample['release_time_definition']['step_h'])
                signature=canonical_digest([asdict(b) for b in batches])
                if signature in known:raise ValueError('Duplicate batch draw; do not silently retry based on results')
                known.add(signature)
                row={a:sample[a] for a in ('schema_version','alpha_definition','release_time_definition','quantity_definition','od_pool')}
                row.update(instance_id=name,configuration=configuration,K=k,deadline_window_alpha=alpha,split='train',development_role=role,
                    instance_seed=seed,path_seed=_find_path_seed(seed+settings['path_seed_offset'],nodes,regions,arcs,batches,tt,lookup),
                    algorithm_seed=settings['collection_seed_base']+index,ccp100_seed=settings['ccp_seed_base']+index,oos5000_seed=None,
                    source_data_sha256=digest(data),source_catalog_digest=claimed,batch_signature=signature,batches=[asdict(b) for b in batches])
                row['instance_digest']=canonical_digest(row);freeze(root/'instances'/f'{name}.json',row)
                records.append({a:row[a] for a in ('instance_id','configuration','K','development_role','algorithm_seed','ccp100_seed','instance_digest','batch_signature')})
                index+=1;print(f'[B20-INSTANCES] {index} {name}',flush=True)
    cat={'protocol_digest':canonical_digest(settings),'source_catalog_digest':claimed,'instances':records,'final_test_opened':False,'oos_scenarios_generated':0}
    cat['catalog_digest']=canonical_digest(cat);freeze(path,cat);return cat
