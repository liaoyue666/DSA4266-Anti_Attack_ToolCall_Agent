"""Build the public, static review payload without source outcomes or labels."""
import hashlib
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/public_workspace_400_v1/annotation_candidates.jsonl'
OUT = ROOT / 'review-web'

def format_value(value, indent=0):
    pad = ' ' * indent
    if isinstance(value, dict):
        lines = []
        for key, child in value.items():
            if isinstance(child, (list,dict)) or '\n' in str(child):
                lines.extend([pad+str(key)+':', format_value(child,indent+2)])
            else:
                lines.append(pad+str(key)+': '+str(child))
        return '\n'.join(lines)
    if isinstance(value, list):
        return '\n'.join(pad+f'记录 {i+1}\n'+format_value(child,indent+2) for i,child in enumerate(value))
    return '\n'.join(pad+line for line in str(value).splitlines())

def readable(content):
    if not yaml or not content:
        return None
    try:
        value = yaml.safe_load(content)
        return format_value(value) if isinstance(value,(dict,list)) else None
    except (yaml.YAMLError, ValueError, RecursionError):
        return None

raw = SOURCE.read_bytes()
rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
assert len(rows) == len({r['candidate_id'] for r in rows}) == 400
cases = [{'candidate_id':r['candidate_id'], 'task_group_id':r['task_group_id'],
          'batch':i//100+1, 'model_input':r['model_input'],
          'observation_views':[readable(o['content']) for o in r['model_input']['observations_before_call']]}
         for i,r in enumerate(rows)]
payload = {'dataset_id':'public-workspace-400-v1','dataset_sha256':hashlib.sha256(raw).hexdigest(),'cases':cases}
OUT.mkdir(exist_ok=True)
(OUT/'dataset.js').write_text('window.REVIEW_DATA = '+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+';\n',encoding='utf-8')
(OUT/'.nojekyll').write_text('',encoding='utf-8')
print(json.dumps({'cases':len(cases),'sha256':payload['dataset_sha256'],'output_bytes':(OUT/'dataset.js').stat().st_size}))
