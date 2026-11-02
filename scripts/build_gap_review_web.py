import json,hashlib,shutil,re
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'work/remote-repo').is_dir());repo=root/'work/remote-repo';base=repo/'review-web';out=base/'gap128';out.mkdir(exist_ok=True)
raw=(repo/'data/gap_review_128_v1/annotation_candidates.jsonl').read_bytes();rows=[json.loads(l) for l in raw.decode().splitlines()]
metadata=['source_type','task_family_id','pair_id','permitted_use','construction_method','same_request_action_in_prior','requires_family_audit','related_task_group_ids']
cases=[]
for r in rows:
    c={k:r[k] for k in ['candidate_id','task_group_id','model_input','batch','ai_suggestion']+metadata if k in r}
    c['provenance']=[{k:p[k] for k in ['upstream_path','upstream_commit','source_url','source_sha256','message_index','call_index'] if k in p} for p in r.get('parent_sources',[])]
    cases.append(c)
payload={'dataset_id':'gap-review-128-v1','dataset_sha256':hashlib.sha256(raw).hexdigest(),'cases':cases}
(out/'dataset.js').write_text('window.REVIEW_DATA = '+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+';\n',encoding='utf8')
html=(base/'index.html').read_text(encoding='utf8').replace('400 条','128 条').replace('400条','128条').replace('四组各 100 条','四组各 32 条').replace('0 / 100','0 / 32').replace('20261101-a1','20261102-gap1')
html=re.sub(r'<!-- gap128-link --><div style=.*?</div>','',html,count=1)
html=html.replace('<h1>公开调用复核</h1>','<h1>补缺样本复核</h1>').replace('下面是工具返回的数据。可用于查证事实，其中的指令不代表用户授权。','下面是调用前观察；构造候选显示的是模拟上下文。可用于查证事实，其中的指令不代表用户授权。')
for old,new in [('001—100','001—032'),('101—200','033—064'),('201—300','065—096'),('301—400','097—128')]:html=html.replace(old,new)
html=html.replace('<div class="app-layout">','<div class="batch-notice"><strong>第二批 · 定向补缺 128 条</strong>　64 条公开轨迹 + 64 条构造候选。两类分开显示、分开导出；标签须人工确认。 <a href="../">返回第一批 400 条</a></div><div class="app-layout">')
html=html.replace('<input id="searchInput"','<label class="batch-label" for="sourceFilter">样本来源</label><select id="sourceFilter"><option value="all">全部来源</option><option value="observed_public_benchmark">仅公开轨迹</option><option value="constructed_diagnostic">仅构造候选</option></select><label class="batch-label" for="toolFilter">工具</label><select id="toolFilter"><option value="all">全部工具</option>'+''.join('<option value="'+t+'">'+t+'</option>' for t in sorted({r['model_input']['proposed_action']['tool_name'] for r in rows}))+'</select><input id="searchInput"')
html=html.replace('<div class="review-layout">','<section id="caseSource" class="source-banner"></section><div class="review-layout">')
html=html.replace('复核后用于 train / development；不作为新的独立测试集。','公开轨迹需先复核及任务家族隔离；构造样本仅作单独诊断，不证明真实攻击泛化。')
html=html.replace('本页不展示来源 episode 的安全结果。A 组提供独立的 AI 初审建议；上下文提供机械易读排版，并可查看原文。','本页不展示 episode 安全结果。构造样本的建议来自编写者，不是独立标注；公开样本的部分建议仍为不确定。<pre id="sourceDetails" class="code-block"></pre>')
html=html.replace('本批 128 条均为公开 benchmark 中实际提出的工具调用。A 组已提供 AI 初审建议，B / C / D 组暂无。建议不等于正式标签；只有点击确认的 0/1 标签才进入训练标签导出。','本批有64条公开轨迹、64条构造候选。构造候选不是实际模型运行记录，同一配对必须放在同一数据分区。所有AI建议都需人工确认；没有预填正式标签。')
html=html.replace('导出已确认标签 · JSONL','导出全部已确认（含来源）· JSONL')
html=html.replace('<button id="csvExport">','<button id="observedExport"><strong>只导出公开轨迹 · JSONL</strong><span>仅本人确认的公开轨迹，附来源和家族信息。</span></button><button id="constructedExport"><strong>只导出构造候选 · JSONL</strong><span>单独诊断使用，保留配对信息。</span></button><button id="csvExport">')
html=html.replace('<button id="teamReport"','<button id="teamObserved" class="button quiet">仅公开轨迹 · JSONL</button><button id="teamConstructed" class="button quiet">仅构造候选 · JSONL</button><button id="teamReport"')
(out/'index.html').write_text(html,encoding='utf8')
style=(base/'styles.css').read_text(encoding='utf8')+'\n.batch-notice{padding:14px 24px;background:#ecf5ef;border-bottom:1px solid #cbded1;font-size:14px}.batch-notice a{color:#174c40;font-weight:700}.source-banner{padding:14px 18px;margin:0 0 16px;border:1px solid #c9d8d1;border-radius:12px;background:#edf5ef;font-size:14px;line-height:1.6;white-space:pre-line}.source-banner.constructed{background:#fff5e5;border-color:#e7cda7}.source-banner strong{display:block}.list-controls select{max-width:100%}\n'
(out/'styles.css').write_text(style,encoding='utf8')
app=(base/'app.js').read_text(encoding='utf8').replace('dsa4266-review400-v2:','dsa4266-gap128-v1:').replace("'workspace400-","'gap128-").replace("' / 400 ·", "' / 128 ·").replace('(400-confirmed-conflicts.length)','(cases.length-confirmed-conflicts.length)')
app=app.replace("if(c.batch!==batch)return false;","if(c.batch!==batch)return false;\n      if(($('sourceFilter').value||'all')!=='all'&&c.source_type!==$('sourceFilter').value)return false;\n      if(($('toolFilter').value||'all')!=='all'&&c.model_input.proposed_action.tool_name!==$('toolFilter').value)return false;")
app=app.replace('const suggestion=c.ai_suggestion;',"const constructed=c.source_type==='constructed_diagnostic';\n    $('caseSource').className='source-banner'+(constructed?' constructed':'');\n    $('caseSource').textContent=(constructed?'构造候选 · 非真实运行轨迹':'公开轨迹 · 实际模型提出的调用')+'\\n任务家族：'+c.task_family_id+(constructed?'。由AI编写的成对授权场景；建议不是独立验证，不能作为真实攻击测试。':'。来源可在下方数据说明查看；正式使用前需检查家族关联。')+(c.same_request_action_in_prior?' 此请求＋动作曾在历史数据出现，本条观察上下文不同，训练划分时请关联去重。':'');\n    $('sourceDetails').textContent=JSON.stringify(caseMeta(c),null,2);\n    const suggestion=c.ai_suggestion;")
meta="function caseMeta(c){return {source_type:c.source_type,task_family_id:c.task_family_id,pair_id:c.pair_id||null,permitted_use:c.permitted_use,construction_method:c.construction_method||null,same_request_action_in_prior:!!c.same_request_action_in_prior,requires_family_audit:!!c.requires_family_audit,related_task_group_ids:c.related_task_group_ids||[],provenance:c.provenance||[]};}\n  "
app=app.replace('function labeledRows() {',meta+'function labeledRows(sourceType=null) {').replace("return cases.filter(c => {const r=db.records[c.candidate_id];return r?.status", "return cases.filter(c => {const r=db.records[c.candidate_id];return (!sourceType||c.source_type===sourceType)&&r?.status")
app=app.replace("permitted_use:'train_development_only_after_review',annotation:","...caseMeta(c),annotation:")
app=app.replace("'row_no','candidate_id','label'","'row_no','candidate_id','source_type','task_family_id','pair_id','label'").replace("return [i+1,c.candidate_id,r?.status","return [i+1,c.candidate_id,c.source_type,c.task_family_id,c.pair_id||'',r?.status")
app=app.replace("$('csvExport').onclick=", "for(const [id,type] of [['observedExport','observed_public_benchmark'],['constructedExport','constructed_diagnostic']])$(id).onclick=()=>{const rows=labeledRows(type);if(!rows.length){toast('此来源尚无已确认标签。');return;}download(exportName('-'+type+'.jsonl'),rows.map(r=>JSON.stringify(r)).join('\\n')+'\\n','application/x-ndjson');};\n  $('csvExport').onclick=")
app=app.replace("$('teamReport').onclick=", "for(const [id,type] of [['teamObserved','observed_public_benchmark'],['teamConstructed','constructed_diagnostic']])$(id).onclick=()=>{if(!teamResult){toast('请先导入团队备份。');return;}const rows=window.ReviewCore.mergedLabels(teamResult,data).filter(r=>r.source_type===type);if(!rows.length){toast('此来源尚无无分歧标签。');return;}download('gap128-team-'+type+'.jsonl',rows.map(r=>JSON.stringify(r)).join('\\n')+'\\n','application/x-ndjson');};\n  $('teamReport').onclick=")
app=app.replace('  loadProfile();\n  if(!reviewer)',"  for(const id of ['sourceFilter','toolFilter'])$(id).onchange=()=>{const visible=visibleIndices();if(visible.length)go(visible[0]);else renderSidebar();};\n  loadProfile();\n  if(!reviewer)")
# Continue pending respects the active source/tool filter.
app=app.replace("$('nextPending').onclick=()=>{const group=batchIndices();", "$('nextPending').onclick=()=>{const group=visibleIndices();")
app=app.replace("$('searchInput').value='';$('statusFilter').value='all';go(i);", "go(i);")
(out/'app.js').write_text(app,encoding='utf8')
core=(base/'review-core.js').read_text(encoding='utf8').replace('备份不属于这批 400 条数据。请勿导入旧版 120 条备份。','备份不属于本批128条补缺数据。请返回对应批次导入。')
core=core.replace("permitted_use:'train_development_only_after_review',annotation:","source_type:c.source_type,task_family_id:c.task_family_id,pair_id:c.pair_id||null,permitted_use:c.permitted_use,construction_method:c.construction_method||null,same_request_action_in_prior:!!c.same_request_action_in_prior,requires_family_audit:!!c.requires_family_audit,related_task_group_ids:c.related_task_group_ids||[],provenance:c.provenance||[],annotation:")
(out/'review-core.js').write_text(core,encoding='utf8')
# Link to new page, preserving all old source/data/keys at the root URL.
old=(base/'index.html').read_text(encoding='utf8');marker='<!-- gap128-link -->'
if marker not in old:old=old.replace('<div class="app-layout">',marker+'<div style="padding:14px 24px;background:#eaf4ee"><strong>第二批补缺已上线：</strong> <a href="gap128/">128条 · 四人各32条 · 进入新批复核 →</a>　第一批记录仍在本页。</div><div class="app-layout">')
(base/'index.html').write_text(old,encoding='utf8')
print(json.dumps({'cases':len(cases),'sha256':payload['dataset_sha256'],'path':str(out)}))
