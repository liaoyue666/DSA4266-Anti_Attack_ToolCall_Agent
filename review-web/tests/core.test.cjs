const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),crypto=require('node:crypto');
const core=require('../review-core.js');
const ctx={window:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../dataset.js'),'utf8'),ctx);const data=ctx.window.REVIEW_DATA;
const time='2026-10-31T01:00:00Z';
function backup(reviewer,batch,label=0){return {schema_version:2,dataset_id:data.dataset_id,dataset_sha256:data.dataset_sha256,reviewer,exported_at:time,records:Object.fromEntries(data.cases.filter(c=>c.batch===batch).map(c=>[c.candidate_id,{reviewer,label,decision:String(label),status:'reviewed',updated_at:time,ai_assisted:false}]))};}
test('400 original inputs match, exactly four disjoint groups, no source outcomes or labels',()=>{
 const raw=fs.readFileSync(path.join(__dirname,'../../data/public_workspace_400_v1/annotation_candidates.jsonl'));
 assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),data.dataset_sha256);
 const original=raw.toString().trim().split('\n').map(JSON.parse);assert.equal(data.cases.length,400);
 data.cases.forEach((c,i)=>{assert.equal(JSON.stringify(c.model_input),JSON.stringify(original[i].model_input));assert.deepEqual(Object.keys(c).sort(),['batch','candidate_id','model_input','observation_views','task_group_id']);});
 for(let i=1;i<=4;i++)assert.equal(data.cases.filter(c=>c.batch===i).length,100);
});
test('four separate reviewers merge to 400 exact input rows',()=>{
 const result=core.mergeBackups([1,2,3,4].map(i=>backup('person'+i,i,i%2)),data);
 const rows=core.mergedLabels(result,data);assert.equal(rows.length,400);assert.equal(new Set(rows.map(r=>r.candidate_id)).size,400);
 assert.equal(rows[0].annotation.reviewer_count,1);assert.equal(rows[0].annotation.ai_assisted,false);
 assert.equal(JSON.stringify(rows[0].model_input),JSON.stringify(data.cases[0].model_input));
});
test('disagreement or uncertainty excluded; same-label second review preserved',()=>{
 const a=backup('a',1),b=backup('b',1,1),id=data.cases[0].candidate_id;
 assert.equal(core.mergedLabels(core.mergeBackups([a,b],data),data).length,0);
 b.records[id].label=0;b.records[id].decision='0';let merged=core.mergeBackups([a,b],data);
 assert.equal(core.mergedLabels(merged,data).length,1);assert.equal(merged.rows[0].reviews.length,2);
 b.records[id].status='disputed';b.records[id].decision='uncertain';b.records[id].label=null;
 assert.equal(core.mergedLabels(core.mergeBackups([a,b],data),data).length,0);
});
test('newer same-person draft invalidates old confirmation regardless of file order',()=>{
 const old=backup('a',1),fresh=structuredClone(old),id=data.cases[0].candidate_id;
 fresh.exported_at='2026-10-31T02:00:00Z';fresh.records[id].status='pending';fresh.records[id].label=null;
 for(const list of [[old,fresh],[fresh,old]])assert.equal(core.mergedLabels(core.mergeBackups(list,data),data).length,99);
 const ambiguous=structuredClone(old);ambiguous.records[id].decision='1';ambiguous.records[id].label=1;
 assert.throws(()=>core.mergeBackups([old,ambiguous],data));
});
test('foreign dataset, spoofed row reviewer, invalid label and timestamp rejected',()=>{
 const b=backup('a',1),id=data.cases[0].candidate_id;
 for(const patch of [{schema_version:1},{dataset_id:'other'},{dataset_sha256:'wrong'},{reviewer:''},{exported_at:'bad'}])assert.throws(()=>core.validateBackup({...b,...patch},data));
 for(const patch of [{reviewer:'someone-else'},{label:1},{label:true},{updated_at:'bad'},{ai_assisted:'false'}]){const bad=structuredClone(b);Object.assign(bad.records[id],patch);assert.throws(()=>core.validateBackup(bad,data));}
});
