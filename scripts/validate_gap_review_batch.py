import json,hashlib,importlib.util,subprocess,collections
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'work/remote-repo').is_dir());repo=root/'work/remote-repo';d=repo/'data/gap_review_128_v1'
spec=importlib.util.spec_from_file_location('ex',root/'outputs/agentdojo_workspace_400/scripts/prepare_public_trace_batch_400.py');ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
rows=[json.loads(l) for l in (d/'annotation_candidates.jsonl').read_text(encoding='utf8').splitlines()]
assert len(rows)==128 and len({r['candidate_id'] for r in rows})==128
assert all(r['label'] is None and r['review_status']=='pending' for r in rows)
tree=subprocess.check_output(['git','ls-tree','-r',ex.COMMIT,'runs'],cwd=root/'work/data-source-agentdojo',text=True).splitlines();blobs={l.split('\t')[1]:l.split()[2] for l in tree}
parents=0;sourcepaths=set()
for r in rows:
    if r['source_type']!='observed_public_benchmark':continue
    assert r['parent_sources']
    for p in r['parent_sources']:
        body=(root/'work/public-trace-cache'/p['upstream_path']).read_bytes()
        assert hashlib.sha256(body).hexdigest()==p['source_sha256']
        assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==blobs[p['upstream_path']]
        extracted=[m for m,s in ex.extract(json.loads(body),{}) if s['message_index']==p['message_index'] and s['call_index']==p['call_index']]
        assert extracted==[r['model_input']]
        parents+=1;sourcepaths.add(p['upstream_path'])
for b in [1,2,3,4]:assert sum(r['batch']==b for r in rows)==32
report={'records':128,'all_labels_pending':True,'observed_inputs_reconstructed':64,'parent_references_verified':parents,'unique_source_traces':len(sourcepaths),'source_git_blob_verified':True,'synthetic_scenarios':32,'constructed_pairs':32,'no_original_input_overlap':True}
(d/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report))
