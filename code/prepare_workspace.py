"""Create the original relative stage layout with adapted tools and text fixtures."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workdir',type=Path,required=True);a=p.parse_args()
REL=Path(__file__).resolve().parents[1];w=a.workdir.resolve()
if w.exists():raise SystemExit('Destination exists; move it aside before preparing a fresh work directory.')
w.mkdir(parents=True)
for src,dst in [('s2','S2'),('s3','S3'),('push','S3b'),('s6','S6')]:
 shutil.copytree(REL/'code'/src,w/dst/'tools')
 for sub in ['inputs','logs','reference','checkpoints','vectors','layers']:(w/dst/sub).mkdir(exist_ok=True)
for sub in ['inputs','S1/logs']:(w/sub).mkdir(parents=True,exist_ok=True)
def read(n):return json.loads((REL/'data'/n).read_text())
def put(n,x):(w/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
fixtures=read('scenarios.json')
for name,x in [('S1/prompts.json',read('s1_prompts.json')),('S1/residual_norms.json',read('residual_norms.json')),('inputs/residual_norms.json',read('residual_norms.json')),('S2/frame_candidates.json',read('s2_frame_candidates.json')),('S3/scenarios_source.json',fixtures['dev']),('S3/validation_prompts_source.json',read('validation_prompts.json')),('S3/arms.json',read('s3_arms.json')),('S3b/scenarios.json',fixtures['dev']),('S3b/pairs_source.json',read('s3b_pairs_source.json')),('S3b/pairs.json',fixtures['s3b_pairs']),('S3b/arms.json',read('s3b_arms.json')),('S6/scenarios_dev.json',fixtures['dev']),('S6/scenarios_heldout.json',fixtures['heldout']),('S6/pairs.json',fixtures['s6_pairs']),('S6/split.json',read('s6_split.json')),('S6/prompt_lines.json',read('s6_prompt_lines.json')),('S6/token_group_plan.json',read('s6_token_group_plan.json'))]:put(name,x)
for stage in ['S2','S3','S3b','S6']:shutil.copy2(REL/'preregistration'/stage/'RULE.md',w/stage/'RULE.md')
print('Text fixtures and tools prepared. Tokenize locally, generate numerical inputs, validate, and freeze before inference.')
