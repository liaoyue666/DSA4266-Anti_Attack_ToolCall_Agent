const {app}=require('./gap128.test.cjs');
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const core=require('../gap128/review-core.js');
test('128 rows, 64+64 sources, four balanced groups, no auto-confirmation',()=>{
 const a=app();assert.equal(a.data.cases.length,128);assert.equal(a.api.labeledRows().length,0);
 for(let b=1;b<=4;b++){
  const rows=a.data.cases.filter(c=>c.batch===b);assert.equal(rows.length,32);assert.equal(rows.filter(c=>c.source_type==='constructed_diagnostic').length,16);
  assert.equal(new Set(rows.filter(c=>c.source_type==='constructed_diagnostic').map(c=>c.model_input.proposed_action.tool_name)).size,8);
 }
 assert.equal(a.elements.has('reasonInput'),false);assert.equal(a.elements.has('evidenceInput'),false);
});
test('all 8 tools have 4 proposed positives and 4 negatives in constructed pairs',()=>{
 const rows=app().data.cases.filter(c=>c.source_type==='constructed_diagnostic');
 for(const tool of new Set(rows.map(c=>c.model_input.proposed_action.tool_name))){
  for(const label of [0,1])assert.equal(rows.filter(c=>c.model_input.proposed_action.tool_name===tool&&c.ai_suggestion.suggested_label===label).length,4);
 }
 for(const id of new Set(rows.map(c=>c.pair_id))){const pair=rows.filter(c=>c.pair_id===id);assert.equal(pair.length,2);assert.deepEqual(pair[0].model_input.observations_before_call,pair[1].model_input.observations_before_call);assert.equal(pair[0].model_input.original_user_request,pair[1].model_input.original_user_request);assert.equal(pair[0].batch,pair[1].batch);}
});
test('source filter works; confirmed exports preserve provenance and split sources',()=>{
 const a=app();a.elements.get('sourceFilter').value='constructed_diagnostic';a.elements.get('sourceFilter').onchange();assert.equal(a.elements.get('caseList').children.length,16);assert.match(a.elements.get('caseSource').textContent,/构造候选/);
 a.choose('0');a.click('confirmButton');const c=a.api.labeledRows('constructed_diagnostic');assert.equal(c.length,1);assert.equal(c[0].source_type,'constructed_diagnostic');assert.ok(c[0].pair_id);assert.equal(c[0].permitted_use,'synthetic_diagnostic_only_after_review');assert.equal(a.api.labeledRows('observed_public_benchmark').length,0);
 a.elements.get('sourceFilter').value='observed_public_benchmark';a.elements.get('sourceFilter').onchange();a.choose('1');a.click('confirmButton');const p=a.api.labeledRows('observed_public_benchmark');assert.equal(p.length,1);assert.ok(p[0].provenance[0].source_url);assert.equal(a.api.labeledRows().length,2);
 assert.match(a.api.exportCsv(),/source_type/);
});
test('tool filter and group bounds; old progress is unchanged and old backups rejected',()=>{
 const storage=new Map([['dsa4266-review400-reviewer','tester'],['dsa4266-review400-v2:old:tester','KEEP_OLD']]);const a=app(storage,'?batch=4');assert.equal(a.api.backupObject().cursor,96);
 a.elements.get('toolFilter').value='share_file';a.elements.get('toolFilter').onchange();assert.ok(a.elements.get('caseList').children.length>=2);a.choose('1');a.click('confirmButton');assert.equal(storage.get('dsa4266-review400-v2:old:tester'),'KEEP_OLD');
 assert.throws(()=>core.validateBackup({...a.api.backupObject(),dataset_id:'public-workspace-400-v1'},a.data));
 const b=app(storage,'?batch=4');assert.equal(b.api.labeledRows().length,1);b.choose('uncertain');b.click('confirmButton');assert.ok(Object.values(b.api.backupObject().records).some(r=>r.status==='disputed'));
});
test('team merge preserves source provenance, pair/family IDs and rejects cross-batch files',()=>{
 const a=app();const c=a.data.cases.find(c=>c.source_type==='constructed_diagnostic');const p=a.data.cases.find(c=>c.source_type==='observed_public_benchmark');
 const backup=(who,row,label)=>({schema_version:2,dataset_id:a.data.dataset_id,dataset_sha256:a.data.dataset_sha256,reviewer:who,exported_at:'2026-11-02T00:00:00Z',records:{[row.candidate_id]:{reviewer:who,status:'reviewed',label,decision:String(label),updated_at:'2026-11-02T00:00:00Z',ai_assisted:true}}});
 const merged=core.mergedLabels(core.mergeBackups([backup('a',c,0),backup('b',p,1)],a.data),a.data);assert.equal(merged.length,2);
 assert.equal(merged.find(r=>r.candidate_id===c.candidate_id).pair_id,c.pair_id);assert.ok(merged.find(r=>r.candidate_id===p.candidate_id).provenance.length);
 assert.throws(()=>core.mergeBackups([{...backup('c',p,0),dataset_sha256:'wrong'}],a.data));
});
