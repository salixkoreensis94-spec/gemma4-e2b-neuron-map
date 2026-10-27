"""3.5h cumulative GPU wall budget; always leave DONE_B when this job exits."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from b_common import O,save,load,freeze_check,RULE_HASH
def main():
 freeze_check();ledger=O/'logs/gpu_budget_b.jsonl'
 prior=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
 used=sum(r['seconds'] for r in prior if r['event']=='end')
 remaining=12600-used;assert remaining>0
 start=time.monotonic();event=dict(event='start',time=time.time(),pid=os.getpid(),prior_seconds=used,remaining_seconds=remaining)
 with ledger.open('a') as f:f.write(json.dumps(event)+'\n')
 status='incomplete';reason='not started';code=1
 try:
  env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
  with (O/f'logs/gpu_b_run_{len(prior):02}.log').open('x') as f:
   p=subprocess.run([sys.executable,str(O/'tools/b_compute.py')],cwd=O.parent,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
  seconds=time.monotonic()-start
  with ledger.open('a') as f:f.write(json.dumps(dict(event='end',time=time.time(),seconds=seconds,returncode=p.returncode))+'\n')
  assert p.returncode==0,f'GPU pipeline exited {p.returncode}; inspect GPU log'
  reason='GPU pipeline complete; reporting'
  for script in ['b_report.py','b_verify.py']:
   with (O/f'logs/{script[:-3]}.log').open('x') as f:
    subprocess.run([sys.executable,str(O/'tools'/script)],cwd=O.parent,stdout=f,stderr=subprocess.STDOUT,check=True)
  status='complete';reason='Full frozen grid, held-out run, reports and output verification complete';code=0
 except subprocess.TimeoutExpired:
  reason='3.5h cumulative GPU wall hard stop; coverage incomplete'
  with ledger.open('a') as f:f.write(json.dumps(dict(event='end',time=time.time(),seconds=time.monotonic()-start,returncode=124))+'\n')
 except Exception as ex:
  reason=str(ex);traceback.print_exc()
 finally:
  result=dict(status=status,phase='B',reason=reason,rule_sha256=RULE_HASH,time=time.time(),gpu_budget_seconds=sum(json.loads(s)['seconds'] for s in ledger.read_text().splitlines() if json.loads(s)['event']=='end'))
  save(O/'logs/DONE_B',result)
  print(json.dumps(result),flush=True)
 return code
if __name__=='__main__':sys.exit(main())
