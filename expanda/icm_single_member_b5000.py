#!/usr/bin/env python3
"""Deploy one frozen ICM shared-supervised member on the original 27 B5000 pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
import tempfile

SHARED = "ICM_SHARED_SUPERVISED"
SINGLE = "ICM_SHARED_SINGLE_991001"
TRAIN_SEED = 991001
BUDGETS = (400, 1000, 5000)


def load_module(exp):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    spec = importlib.util.spec_from_file_location("screen27", exp/"ppo_joint_b5000_screen27_v1/run.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    return s


def read(path):
    return json.loads(path.read_bytes())


def freeze(s, path, value):
    s.freeze(path, value)


def prepare(s, exp, diag, out):
    old = diag/"ccp_component_joint_b5000_v1"
    plan = read(old/"PLAN.json")
    if s.canonical({k:v for k,v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("Source plan checksum mismatch")
    s.p.verify(plan["files"])
    settings = plan["settings"]
    assert (settings["budget"], settings["population"], settings["scenario_count"], settings["pool_size"]) == (5000, 250, 100, 8)
    assert settings["screen_seeds"] == [993001,993002,993003] and tuple(settings["milestones"]) == BUDGETS
    assert sorted(r["configuration"] for r in plan["screen"]) == [f"S{i}" for i in range(9)]
    train = Path(plan["source"])/"shared_model"
    srcbundle = train/"BUNDLE.json"
    if s.digest(srcbundle) != plan["files"][str(srcbundle)]:
        raise ValueError("Source bundle differs from the evaluated ensemble")
    bundle = read(srcbundle)
    epoch = bundle["selected"]["epoch"]
    weights = f"seed_{TRAIN_SEED}_epoch_{epoch}.pt"
    if len(bundle["checkpoints"]) != 3 or weights not in bundle["checkpoints"]:
        raise ValueError("Expected the original three members and predeclared first training seed")
    member = out/"one_member"
    member.mkdir(exist_ok=True)
    weight_dst = member/weights
    if not weight_dst.exists():
        shutil.copy2(train/weights,weight_dst)
    if s.digest(weight_dst) != bundle["checkpoint_sha256"][weights]:
        raise ValueError("Single member checksum mismatch")
    # All original feature scalers, task gates and legal-mask handling stay in the bundle.
    freeze(s,member/"BUNDLE.json", dict(bundle,checkpoints=[weights],checkpoint_sha256={weights:s.digest(weight_dst)}))
    evidence = diag/"ccp_paper_evidence_b5000_v1"
    fronts = evidence/"DEV_FRONTS.jsonl.gz"
    if read(evidence/"PROVENANCE.json")["files"][str(old/"PLAN.json")] != s.digest(old/"PLAN.json"):
        raise ValueError("Archived development fronts originate from a different comparison")
    meta = dict(original=str(old),original_plan=plan["sha256"],source_model=s.digest(srcbundle),
                single_model=s.digest(member/"BUNDLE.json"),training_seed=TRAIN_SEED,epoch=epoch,
                expanda=str(exp),fronts=str(fronts),fronts_sha256=s.digest(fronts),
                program_sha256=s.digest(Path(__file__)),seeds=settings["screen_seeds"])
    freeze(s,out/"PLAN.json",meta)
    return plan


def job_path(out, name, seed):
    return out/"runs"/name/f"seed_{seed}"


def worker(s, out, name, seed):
    from search_b5000 import run_search
    meta = read(out/"PLAN.json")
    original = Path(meta["original"])
    plan = read(original/"PLAN.json")
    entry = next(e for e in plan["screen"] if e["instance_id"] == name)
    if seed not in meta["seeds"] or plan["sha256"] != meta["original_plan"]:
        raise ValueError("Undeclared run")
    control = read(original/"screen"/name/f"seed_{seed}"/SHARED/"RUN_COMPLETE.json")
    s.audit.validate_run(control,entry,seed,plan["settings"])
    if control["model_bundle_sha256"] != meta["source_model"] or control["plan_sha256"] != plan["sha256"]:
        raise ValueError("Wrong reference run")
    dest = job_path(out,name,seed)
    dest.parent.mkdir(parents=True,exist_ok=True)
    instance = original/"instances"/(name+".json")
    if read(instance) != entry:
        raise ValueError("Instance changed")
    member = out/"one_member"
    task = s.scheduler.task(plan,name,seed,SHARED,"screen",member)
    if task["bundle_sha256"] != meta["single_model"]:
        raise ValueError("Single member bundle changed")
    def valid(row):
        s.audit.validate_run(row,entry,seed,plan["settings"],control["scenario_digest"])
        if (row["model_bundle_sha256"] != meta["single_model"] or row["plan_sha256"] != s.canonical(meta)
                or row["task_sha256"] != s.canonical(task) or row["arm"] != SHARED):
            raise ValueError("Wrong single-member run")
    with (dest.parent/(dest.name+".lock")).open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if dest.exists():
            valid(read(dest/"RUN_COMPLETE.json"))
            return
        temp = Path(tempfile.mkdtemp(prefix=dest.name+".attempt-",dir=dest.parent))
        # The search arm stays SHARED internally to use the ensemble's exact controller.
        row = run_search(instance,plan["settings"],task,temp,s.canonical(meta))
        valid(row)
        temp.rename(dest)


def report(s, plan, out):
    meta = read(out/"PLAN.json")
    keys = {(e["instance_id"],n) for e in plan["screen"] for n in meta["seeds"]}
    originals = {}
    with gzip.open(meta["fronts"],"rt",encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals.setdefault(row["method"],[]).append(row)
    if len(originals)!=11 or any(len(rows)!=27 or {(r["instance"],r["seed"]) for r in rows}!=keys for rows in originals.values()):
        raise ValueError("Incomplete eleven-method evidence")
    groups = {}
    original=Path(meta["original"])
    for method in ("BASE",SHARED):
        groups[method]=[read(original/"screen"/i/f"seed_{n}"/method/"RUN_COMPLETE.json") for i,n in sorted(keys)]
        archive = {(r["instance"],r["seed"]):r for r in originals[method]}
        for r in groups[method]:
            a=archive[r["instance"],r["seed"]]
            if (a["instance_digest"],a["scenario_digest"],a["budget_fronts"]) != (r["instance_digest"],r["scenario_digest"],r["budget_fronts"]):
                raise ValueError("Reference fronts do not match development archive")
    groups[SINGLE]=[read(job_path(out,i,n)/"RUN_COMPLETE.json") for i,n in sorted(keys)]
    originals.update(groups)
    normalized,bounds = s.audit.union_normalize(originals)
    freeze(s,out/"INSTANCE_BOUNDS.json",bounds)
    summary,per_run=[],[]
    for b in BUDGETS:
        at = {m:[dict(r,metric=next(x["hv"] for x in r["budget_fronts"] if x["evaluation"]==b)) for r in normalized[m]] for m in groups}
        for method,ref in (("BASE","BASE"),(SHARED,"BASE"),(SINGLE,"BASE"),(SINGLE,SHARED)):
            own=at[method]
            v=s.p.paired(own,at[ref],field="metric")
            lo,hi=s.p.hv.interval(v["per_run"],"percent",3000)
            def times(key):
                values=[r[key] for r in own]
                return (mean(values),stdev(values)) if b==5000 else ("","")
            wall,sd=times("optimisation_seconds")
            infer,infer_sd=times("inference_seconds")
            summary.append(dict(评价次数=b,方法=method,参照=ref,配对数=v["pairs"],平均HV=v["mean_hv"],
                平均配对提升百分比=v["mean_paired_percent"],提升区间下限=lo,提升区间上限=hi,
                平均HV差=v["mean_absolute_difference"],HV差区间下限=v["absolute_ci_low"],HV差区间上限=v["absolute_ci_high"],
                胜=v["wins"],平=v["ties"],负=v["losses"],平均搜索秒=wall,搜索秒标准差=sd,平均推理秒=infer,推理秒标准差=infer_sd))
            own_runs = {(r["instance"],r["seed"]):r for r in own}
            ref_runs = {(r["instance"],r["seed"]):r for r in at[ref]}
            for pair in v["per_run"]:
                key=pair["instance"],pair["seed"]
                current,reference=own_runs[key],ref_runs[key]
                per_run.append(dict(method=method,reference_method=ref,evaluation=b,
                    optimisation_seconds=current["optimisation_seconds"] if b==5000 else None,
                    reference_seconds=reference["optimisation_seconds"] if b==5000 else None,
                    inference_seconds=current["inference_seconds"] if b==5000 else None,
                    node=current.get("hostname"),reference_node=reference.get("hostname"),**pair))
    s.p.csv_write(out/"RESULTS.csv",summary)
    s.p.csv_write(out/"PER_PAIR.csv",per_run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanda",type=Path,default=Path.cwd())
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    parser.add_argument("--worker",nargs=2,metavar=("INSTANCE","SEED"),help=argparse.SUPPRESS)
    args=parser.parse_args()
    exp=args.expanda.resolve()
    diag=exp/"transformer_ppo_runs/diagnostics"
    out=diag/"ccp_icm_shared_single_991001_b5000_v1"
    s=load_module(exp)
    if args.worker:
        worker(s,out,args.worker[0],int(args.worker[1]))
        return
    out.mkdir(parents=True,exist_ok=True)
    with (out/".launch.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(s,exp,diag,out)
        logs=out/"logs"
        logs.mkdir(exist_ok=True)
        def run(name,seed):
            log=logs/f"{name}_{seed}.log"
            cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--expanda",str(exp),"--worker",name,str(seed)]
            with log.open("a") as stream:
                completed=subprocess.run(cmd,cwd=exp,stdout=stream,stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Worker exited {completed.returncode}. See {log}")
            return name,seed
        jobs=[(e["instance_id"],n) for e in plan["screen"] for n in plan["settings"]["screen_seeds"]]
        print(f"Single member {TRAIN_SEED}, {len(jobs)} paired searches, workers={args.workers}",flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run,*j) for j in jobs]
            try:
                for i,f in enumerate(as_completed(futures),1):
                    print(f"{i}/{len(jobs)}",*f.result(),flush=True)
            finally:
                for f in futures: f.cancel()
        report(s,plan,out)
        print("COMPLETE",out/"RESULTS.csv",flush=True)


if __name__=="__main__":
    main()
