#!/usr/bin/env python3
"""Read existing experiments and export paper evidence. Never runs a search or optimizer."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('evidence_screen27', HERE.parent/'ppo_joint_b5000_screen27_v1/run.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p = s.p
read, write, digest = s.read, s.write, s.digest
NAMES = dict(p.hv.NAMES)
NAMES.pop('ICM_TERMINAL_PPO')
NAMES['PPO_PILOT_U001'] = 'ICM PPO u1（最高分选择）'


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return dict(returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as e: return dict(unavailable=str(e))


def optional_text(path):
    try: return Path(path).read_text().strip()
    except OSError: return None


def hardware():
    cpu_command=command(['lscpu','-J'])
    try:cpu={r['field'].rstrip(':'):r['data'] for r in json.loads(cpu_command.get('stdout','{}')).get('lscpu',[])}
    except (ValueError,KeyError,TypeError):cpu={}
    memory = optional_text('/proc/meminfo')
    match = re.search(r'^MemTotal:\s+(\d+)\s+kB', memory or '', re.M)
    status = optional_text('/proc/self/status') or ''
    allowed = {k:re.search(r'^'+k+r':\s*(.*)$',status,re.M).group(1) for k in ('Cpus_allowed_list','Mems_allowed_list') if re.search(r'^'+k+r':',status,re.M)}
    cgroup = optional_text('/proc/self/cgroup'); limits = {}
    if cgroup:
        for line in cgroup.splitlines():
            if line.startswith('0::'):
                folder = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
                for name in ('cpu.max','cpuset.cpus.effective','memory.max','memory.high'):
                    limits[name] = optional_text(folder/name)
    envkeys = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
               'SLURM_JOB_ID','SLURM_CPUS_PER_TASK','SLURM_CPUS_ON_NODE','SLURM_MEM_PER_NODE','SLURM_MEM_PER_CPU','PBS_JOBID','PBS_NP','NSLOTS')
    return dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
        platform=platform.platform(), python=platform.python_version(), numpy=str(p.np.__version__), torch=str(p.torch.__version__),
        process_id=os.getpid(),parent_pid=os.getppid(),cpu_model=cpu.get('Model name'),lscpu=cpu_command,
        cpu_details=cpu,logical_cpus=os.cpu_count(),
        current_cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        total_node_memory_bytes=int(match.group(1))*1024 if match else None,
        current_allowed=allowed,current_cgroup_limits=limits,current_scheduler_environment={k:os.environ[k] for k in envkeys if k in os.environ},
        historical_allocated_cores=None,historical_allocated_memory_bytes=None,
        qualification='Captured now, not a historical allocation record. Node total/affinity/peak RSS are not reserved experiment cores or RAM. Historical allocation is unknown unless scheduler/admin records independently establish it.')


def concurrency(folder):
    log = folder/'pipeline.nohup.log'; values=[]
    if log.exists():
        with log.open(errors='replace') as f:
            for line in f:
                if 'heartbeat' in line:
                    m=re.search(r'\brunning=(\d+)',line)
                    if m: values.append(int(m.group(1)))
    return dict(experiment=str(folder),log=str(log),log_sha256=digest(log) if log.exists() else None,
        heartbeat_observed_max_parallel_workers=max(values) if values else None,
        observations=len(values),qualification='Observed heartbeat maximum; does not establish reserved CPU cores or memory.')


def mean_optional(rows,key):
    return float(statistics.mean(r[key] for r in rows)) if rows and all(r.get(key) is not None for r in rows) else None


def summarize_development(groups, out, draws=3000):
    summary=[]; raw=[]; configurations=[]
    for budget in s.BUDGETS:
        at={m:[dict(r,metric=next(x['hv'] for x in r['budget_fronts'] if x['evaluation']==budget)) for r in records] for m,records in groups.items()}
        for method in NAMES:
            own=at[method]; q=p.paired(own,at['BASE'],field='metric',draws=draws)
            lo,hi=p.hv.interval(q['per_run'],'percent',draws)
            summary.append({'评价次数':budget,'方法':NAMES[method],'方法标识':method,'参照':'Base','运行对数':q['pairs'],
                '平均HV':q['mean_hv'],'HV样本标准差_跨全部运行':statistics.stdev(r['metric'] for r in own),
                '相对Base平均配对提升(%)':q['mean_paired_percent'],'提升95%下限(%)':lo,'提升95%上限(%)':hi,
                '平均HV绝对差':q['mean_absolute_difference'],'绝对差95%下限':q['absolute_ci_low'],'绝对差95%上限':q['absolute_ci_high'],
                '胜Base':q['wins'],'平Base':q['ties'],'负Base':q['losses'],
                '平均B5000搜索秒':mean_optional(own,'optimisation_seconds') if budget==5000 else None})
            raw.extend(dict(method=method,evaluation=budget,**r) for r in q['per_run'])
            for cfg in sorted({r['configuration'] for r in q['per_run']}):
                records=[r for r in q['per_run'] if r['configuration']==cfg]
                configurations.append(dict(method=method,evaluation=budget,configuration=cfg,pairs=len(records),
                    mean_hv=p.equal_mean(records,'own'),percent_vs_base=p.hv.equal_mean(records,'percent'),
                    absolute_vs_base=p.equal_mean(records,'absolute')))
    p.csv_write(out/'DEV_11_METHODS.csv',summary)
    p.csv_write(out/'DEV_ENDPOINT_5000.csv',[r for r in summary if r['评价次数']==5000])
    p.csv_write(out/'DEV_PER_RUN.csv',raw);p.csv_write(out/'DEV_PER_CONFIGURATION.csv',configurations)
    return summary


def validation_report(label, groups, out_rows, per_run):
    base=groups[0]
    for update,rows in sorted(groups.items()):
        v=p.paired(rows,base)
        out_rows.append(dict(version=label,update=update,runs=len(rows),configurations=len({r['configuration'] for r in rows}),
            mean_validation_hv_frozen_training=v['mean_hv'],mean_hv_difference_vs_u0=v['mean_absolute_difference'],
            mean_percent_vs_u0=v['mean_paired_percent'],wins=v['wins'],ties=v['ties'],losses=v['losses'],
            search_seconds_sum=sum(r['optimisation_seconds'] for r in rows),
            qualification='Validation uses the original frozen training normalization, not development union bounds.'))
        per_run.extend(dict(version=label,update=update,**r) for r in v['per_run'])


def cost_stage(version, stage, paths):
    rows=[read(path) for path in paths]
    if any(not r.get('completed') for r in rows):raise ValueError('Incomplete cost record')
    return dict(version=version,stage=stage,completed_searches=len(rows),
        search_objective_evaluations=sum(r['real_objective_evaluation_count'] for r in rows),
        extra_label_evaluations=sum(r.get('hidden_candidate_objective_evaluations',0) for r in rows),
        cumulative_search_seconds=sum(r['optimisation_seconds'] for r in rows),
        qualification='Sum of completed search durations, not parallel wall time; optimizer/startup/interrupted attempts excluded.')


def export_ppo(old, pilot_root, out):
    table=[]; raw=[]; costs=[]; updates=[]; versions={}
    old_plan=read(old/'PLAN.json'); pilot_plan=read(pilot_root/'PLAN.json')
    old_groups={u:[read(f) for f in sorted((old/f'terminal/validation/u{u:03d}').glob('*/seed_*/RUN_COMPLETE.json'))] for u in (0,4,8)}
    if any(len(v)!=54 for v in old_groups.values()):raise ValueError('Expected 54 old validation runs/checkpoint')
    validation_report('旧版_B5000_8次更新',old_groups,table,raw)
    new_groups={0:p.cached_baselines(pilot_plan,pilot_plan['tuning'])}
    for u in (1,2,3):
        new_groups[u]=[read(f) for f in sorted((pilot_root/f'runs/tuning/u{u:03d}/greedy').glob('*/seed_*/RUN_COMPLETE.json'))]
    if any(len(v)!=9 for v in new_groups.values()):raise ValueError('Expected 9 new tuning runs/checkpoint')
    validation_report('新表_PPO小试v2_3次更新',new_groups,table,raw)
    old_rollouts=sorted((old/'terminal/rollouts').glob('u*/*/RUN_COMPLETE.json'))
    new_rollouts=sorted((pilot_root/'runs/training').glob('u*/*/seed_*/RUN_COMPLETE.json'))
    references=sorted((pilot_root/'runs/training_reference').glob('u*/*/seed_*/RUN_COMPLETE.json'))
    if (len(old_rollouts),len(new_rollouts),len(references))!=(32,9,9):raise ValueError('Unexpected training search counts')
    costs.extend([cost_stage('old','PPO_training',old_rollouts),cost_stage('pilot_v2','PPO_training',new_rollouts),
                  cost_stage('pilot_v2','paired_shared_training_references',references)])
    for label,root,pattern in [('old',old,'terminal/validation/u*/*/seed_*/RUN_COMPLETE.json'),
                               ('pilot_v2',pilot_root,'runs/tuning/u*/greedy/*/seed_*/RUN_COMPLETE.json'),
                               ('pilot_v2_confirmation',pilot_root,'runs/confirmation/u*/*/*/seed_*/RUN_COMPLETE.json'),
                               ('pilot_v2_replay',pilot_root,'runs/reproduction/u*/*/*/seed_*/RUN_COMPLETE.json')]:
        costs.append(cost_stage(label,'validation_or_confirmation_or_replay',sorted(root.glob(pattern))))
    for label,root,sub,n in [('old',old,'terminal/checkpoints',8),('pilot_v2',pilot_root,'checkpoints',3)]:
        metrics=[]
        for u in range(1,n+1):
            path=root/sub/f'update_{u:03d}/UPDATE.json'; data=read(path)
            metrics.append(dict(update=u,metrics=data,source=str(path),sha256=digest(path)))
            updates.append(dict(version=label,update=u,optimizer_steps=data.get('optimizer_steps'),kl_early_stop=data.get('kl_early_stop'),
                actor_parameter_l2_change=data.get('actor_parameter_l2_change'),actions=data.get('actions'),
                complete_episodes=data.get('complete_episodes',len(data.get('reward_pairs',[]))),optimizer_seconds=None))
        versions[label]=dict(updates=metrics,optimizer_time_qualification='Standalone optimizer duration was not saved; do not infer it from file timestamps.')
    versions['old']['selection']=read(old/'terminal/SELECTED.json')
    versions['pilot_v2']['selection']=read(pilot_root/'SELECTION.json')
    versions['new_table_deployed_checkpoint']=str(pilot_root/'checkpoints/update_001')
    versions['new_table_training']=dict(updates=3,policy_searches=9,paired_reference_searches=9,tuning_updates=[0,1,2,3],
        configurations=pilot_plan['pilot']['configurations'],deployed_update=1,
        policy_training_episodes_before_deployed_checkpoint=3,paired_references_before_deployed_checkpoint=3,
        qualification='u1 is best among updated checkpoints; u0 was best including initialization. u4/u8 belong only to the old experiment.')
    p.csv_write(out/'PPO_VALIDATION.csv',table);p.csv_write(out/'PPO_VALIDATION_PER_RUN.csv',raw)
    p.csv_write(out/'PPO_UPDATES.csv',updates);p.csv_write(out/'PPO_SEARCH_COSTS.csv',costs)
    write(out/'PPO_PROVENANCE.json',versions)


def export_supervision(root, independent, out):
    entries=read(root/'instances/catalog.json')['instances']; rows=[]; components=[]
    for e in entries:
        for arm in ('COLLECT_I','COLLECT_CM'):
            path=root/'collection'/e['instance_id']/arm/'RUN_COMPLETE.json'; v=read(path)
            if not v.get('completed'):raise ValueError('Incomplete collection')
            rows.append(dict(instance=e['instance_id'],configuration=e['configuration'],role=e['development_role'],arm=arm,
                search_evaluations=v['real_objective_evaluation_count'],extra_label_evaluations=sum(v['audit_evaluations'].values()),
                search_seconds=v['optimisation_seconds'],audit_seconds=v['audit_seconds']))
            for c in ('I' if arm=='COLLECT_I' else 'CM'):
                components.append(dict(instance=e['instance_id'],configuration=e['configuration'],role=e['development_role'],component=c,
                    states=v['audit_states'][c],extra_evaluations=v['audit_evaluations'][c],audit_seconds=v['audit_seconds_by_component'][c]))
    p.csv_write(out/'SUPERVISION_TRAJECTORY_COSTS.csv',rows);p.csv_write(out/'SUPERVISION_LABEL_COSTS.csv',components)
    stages=[]
    for label,folder in [('shared',root/'shared_model'),('independent',independent),('historical_pretraining_diagnostic',root/'pretrain')]:
        path=folder/'BUNDLE.json'
        if path.exists():
            b=read(path);stages.append(dict(model=label,recorded_training_seconds_this_invocation=b.get('training_seconds_this_invocation'),
                separated_optimizer_seconds=None,separated_offline_validation_seconds=None,
                qualification='Recorded fitting invocation includes label/scaler preparation and gate/checkpoint validation; previous resumed invocations may be absent.'))
    p.csv_write(out/'SUPERVISION_FIT_COSTS.csv',stages)
    summary=dict(completed_trajectories=len(rows),search_evaluations=sum(r['search_evaluations'] for r in rows),
        extra_label_evaluations=sum(r['extra_label_evaluations'] for r in rows),
        cumulative_search_seconds=sum(r['search_seconds'] for r in rows),
        interrupted_attempts=len(list((root/'interrupted').glob('*'))),
        qualification='I initialization and shared C/M Base trajectory counted once each. Audit time is contained in collection duration, not an additional wall-time total. Interrupted effort is not included.')
    write(out/'SUPERVISION_COST_SUMMARY.json',summary)


def collect(source,out):
    if not read(source/'STATUS.json').get('completed'):raise ValueError('27-pair comparison must be complete')
    plan=s.checked_plan(source);old=Path(plan['original']);pilot_root=Path(plan['source']);oldplan=read(old/'PLAN.json')
    original=[read(Path(path)) for path in plan['union_pool']]
    new=[read(path) for path in sorted((source/'runs').glob('*/seed_*/PPO_U001/RUN_COMPLETE.json'))]
    if len(original)!=270 or len(new)!=27:raise ValueError('Expected 270 original and 27 updated PPO runs')
    for row in new:s.validate_ppo(row,plan)
    groups={m:[r for r in original if r['arm']==m] for m in s.POOL_ARMS};groups['PPO_PILOT_U001']=new
    if any(len(v)!=27 for v in groups.values()):raise ValueError('Incomplete 11-method panel')
    out.mkdir(parents=True,exist_ok=True)
    normalized,bounds=s.audit.union_normalize(groups)
    table=summarize_development(normalized,out);write(out/'DEV_INSTANCE_BOUNDS.json',bounds)
    with gzip.open(out/'DEV_FRONTS.jsonl.gz','wt',encoding='utf-8') as f:
        for method,records in groups.items():
            for row in records:f.write(json.dumps(dict(method=method,instance=row['instance'],configuration=row['configuration'],seed=row['seed'],
                instance_digest=row['instance_digest'],scenario_digest=row['scenario_digest'],budget_fronts=row['budget_fronts']),ensure_ascii=False)+'\n')
    costs=[]
    for method,records in groups.items():
        costs.append(dict(method=method,runs=len(records),mean_search_seconds=mean_optional(records,'optimisation_seconds'),
            mean_inference_seconds=mean_optional(records,'inference_seconds'),mean_feature_seconds=mean_optional(records,'feature_construction_seconds'),
            mean_objective_seconds=mean_optional(records,'objective_evaluation_seconds'),mean_final_hv_seconds=mean_optional(records,'final_hv_seconds'),
            mean_peak_memory_mib=mean_optional(records,'peak_memory_mib'),
            total_actual_evaluations=sum(r['real_objective_evaluation_count'] for r in records),
            hosts=json.dumps(sorted({r.get('hostname','not recorded') for r in records}))))
    p.csv_write(out/'DEPLOYMENT_COSTS.csv',costs)
    hardware_record=hardware();hardware_record['recorded_search_hosts']=sorted({r.get('hostname','not recorded') for v in groups.values() for r in v})
    hardware_record['concurrency_evidence']=[concurrency(folder) for folder in (old,pilot_root,source,Path(oldplan['source']))]
    write(out/'ENVIRONMENT.json',hardware_record)
    export_ppo(old,pilot_root,out);export_supervision(Path(oldplan['source']),Path(oldplan['independent_model']),out)
    endpoint={r['方法标识']:r for r in table if r['评价次数']==5000};shared=endpoint['ICM_SHARED_SUPERVISED'];simple=endpoint['C_RULE'];ppo=endpoint['PPO_PILOT_U001']
    summary=dict(completed=True,new_searches=0,new_training_updates=0,methods=11,pairs_per_method=27,budgets=list(s.BUDGETS),
        reference='BASE',normalization='same-instance all eleven methods, all seeds and budgets; reference (1.1,1.1,1.1); no /1.331',
        shared_vs_simple_percentage_point_gap=shared['相对Base平均配对提升(%)']-simple['相对Base平均配对提升(%)'],
        shared_minus_simple_seconds=shared['平均B5000搜索秒']-simple['平均B5000搜索秒'],
        proposed_primary_method='ICM_SHARED_SUPERVISED',
        proposed_selection_reason='B5000 remains the primary endpoint. Shared has slightly higher mean vs Base and lower time than updated PPO on the inspected panel; no claim of statistically superior final HV. PPO remains a budget-dependent comparative variant.',
        shared_endpoint=shared,ppo_endpoint=ppo,
        missing=['Historical reserved cores/RAM unless external allocation records exist','Separate offline supervised validation/optimizer timings','Fresh disjoint S0-S11 test results'],
        final_test_opened=False,scope='Development evidence only; one instance/configuration limits bootstrap generalization.')
    write(out/'SUMMARY.json',summary)
    provenance={str(source/'PLAN.json'):digest(source/'PLAN.json'),str(old/'PLAN.json'):digest(old/'PLAN.json'),
                str(pilot_root/'SELECTION.json'):digest(pilot_root/'SELECTION.json')}
    write(out/'PROVENANCE.json',dict(source_experiment=str(source),source_plan_sha256=plan['sha256'],files=provenance))
    shutil.copyfile(HERE/'README_CN.txt',out/'README_CN.txt')
    archive=shutil.make_archive(str(out),'zip',root_dir=out)
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('[PAPER-EVIDENCE] '+archive,flush=True)


def main():
    diag=HERE.parent/'transformer_ppo_runs/diagnostics';parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=diag/'ccp_joint_ppo_b5000_screen27_v1')
    parser.add_argument('--out',type=Path,default=diag/'ccp_paper_evidence_b5000_v1');a=parser.parse_args()
    source=a.source.resolve();out=a.out.resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('Separate report output required')
    collect(source,out)


if __name__=='__main__':main()
