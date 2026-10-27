"""No-model syntax, S6 authored validation, and regression-probe checks."""
import collections,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def main():
 files=sorted((ROOT/'code').rglob('*.py'))
 for p in files:compile(p.read_text(),str(p.relative_to(ROOT)),'exec')
 path=ROOT/'code/s6/style_classifier.py';spec=importlib.util.spec_from_file_location('style_classifier',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 pairs=json.loads((ROOT/'data/scenarios.json').read_text())['s6_pairs'];expected=json.loads((ROOT/'data/classifier_expected.json').read_text())
 result=[]
 for p in pairs:
  if p['split']!='validation':continue
  for side,label in [('positive',p['style']),('negative',p['partner_label'])]:
   result.append(dict(pair=p['id'],side=side,gold=label,prediction=m.classify(p[side])['dominant']))
 assert result==expected['rows'],'Changed validation predictions'
 correct=sum(r['gold']==r['prediction'] for r in result)
 assert (len(result),correct)==(128,126)
 cm={g:{p:0 for p in expected['confusion_matrix']} for g in expected['confusion_matrix']}
 for r in result:cm[r['gold']][r['prediction']]+=1
 assert cm==expected['confusion_matrix']
 probes=json.loads((ROOT/'data/classifier_probes.json').read_text())
 for p in probes:assert m.classify(p['text'])['dominant']==p['expected'],p
 print(json.dumps(dict(python_syntax_files=len(files),classifier_version=m.VERSION,validation_n=len(result),validation_correct=correct,per_label=expected['per_label'],regression_probes=len(probes),passed=True),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
