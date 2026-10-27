import json
from pathlib import Path
from config import ROOT as R
O=R/'S6'
def load(p):return json.loads(p.read_text())
def save(n,x):
 with (O/n).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def render(messages):return ''.join('<|turn>'+('model' if m['role']=='assistant' else m['role'])+'\n'+m['content']+'<turn|>\n' for m in messages)+'<|turn>model\n'
def main():
 rows=[]
 for p in load(O/'pairs.json'):
  if p['split']!='build':continue
  for side in ['prefix','positive','negative']:
   rows.append(dict(id='pair/'+p['id']+'/'+side,text=p['prefix']+(p[side] if side!='prefix' else '')))
 additions=load(O/'prompt_lines.json')['additions']
 for phase in ['dev','heldout']:
  for s in load(O/f'scenarios_{phase}.json'):
   for style in ['none',*additions]:
    messages=[dict(m) for m in s['messages']]
    if style!='none':messages[0]['content']+=' '+additions[style]
    rows.append(dict(id=f'{phase}/{s["id"]}/{style}',text=render(messages)))
 for p in load(R/'S3b/validation_prompts.json'):
  rows.append(dict(id=f'validation/{p["id"]}',text=p['rendered']))
 save('tokenize_tasks.json',rows)
 print('tokenization tasks',len(rows))
if __name__=='__main__':main()
