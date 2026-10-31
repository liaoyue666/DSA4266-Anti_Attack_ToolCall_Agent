(() => {
  'use strict';
  const data = window.REVIEW_DATA;
  const cases = data.cases;
  const ids = new Set(cases.map(c => c.candidate_id));
  const $ = id => document.getElementById(id);
  const prefix = 'dsa4266-review400-v2:' + data.dataset_sha256 + ':';
  const requestedBatch = Number(new URLSearchParams(window.location.search).get('batch'));
  const validBatch = value => [1,2,3,4].includes(Number(value));
  let batch = validBatch(requestedBatch) ? requestedBatch : 1;
  let reviewer = '', db = {records: {}, cursor: 0}, current = 0, storageOK = true, toastTimer;
  try { reviewer = localStorage.getItem('dsa4266-review400-reviewer') || ''; } catch { storageOK = false; }

  function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 4200); }
  function key() { return prefix + encodeURIComponent(reviewer); }
  function persist() {
    db.cursor = current; db.batch = batch;
    if (!reviewer) return;
    try { localStorage.setItem(key(), JSON.stringify(db)); storageOK = true; }
    catch { storageOK = false; }
    $('storageWarning').hidden = storageOK;
    updateSaveStatus();
  }
  function loadProfile() {
    db = {records: {}, cursor: 0};
    try { const saved = localStorage.getItem(key()); if (saved) { const parsed = JSON.parse(saved); window.ReviewCore.validateBackup({schema_version:2,dataset_id:data.dataset_id,dataset_sha256:data.dataset_sha256,reviewer,exported_at:new Date().toISOString(),...parsed},data); db = parsed; } }
    catch { toast('本机进度暂时无法读取。请导入已有备份，或使用新的复核人代号。'); }
    current = Number.isInteger(db.cursor) ? Math.max(0, Math.min(cases.length - 1, db.cursor)) : 0;
    if (!validBatch(requestedBatch) && validBatch(db.batch)) batch=Number(db.batch);
    if (!batchIndices().includes(current)) current=batchIndices()[0];
    $('batchSelect').value=String(batch);
    $('profileButton').textContent = reviewer ? '复核人：' + reviewer : '选择复核人';
    $('storageWarning').hidden = storageOK;
    render();
  }
  function batchIndices(){ return cases.map((c,i)=>({c,i})).filter(({c})=>c.batch===batch).map(({i})=>i); }
  function record() { return db.records[cases[current].candidate_id]; }
  function decision() { return document.querySelector('input[name="decision"]:checked')?.value || null; }
  function statusText(r) { return r?.status === 'reviewed' ? '已确认 · ' + (r.label === 0 ? '授权' : '未授权') : r?.status === 'disputed' ? '不确定 · 待讨论' : r ? '草稿 · 待确认' : '待复核'; }
  function updateSaveStatus() {
    const r = record();
    $('saveStatus').textContent = !storageOK ? '请导出备份保存进度' : r ? (r.status === 'pending' ? '草稿已保存到本机 · 尚未确认' : '复核记录已保存到本机') : '尚未填写';
  }
  function updateProgress() {
    const group=batchIndices();
    const n=group.filter(i=>db.records[cases[i].candidate_id]?.status==='reviewed').length;
    $('reviewedCount').textContent = n + ' / ' + group.length;
    $('progressBar').style.width = (n / group.length * 100) + '%';
    $('batchSummary').textContent=String.fromCharCode(64+batch)+' 组 · '+String(group[0]+1).padStart(3,'0')+'—'+String(group[group.length-1]+1).padStart(3,'0')+' · 本人已确认 '+n+' / '+group.length;
    const r = record();
    $('caseStatus').textContent = statusText(r);
    $('caseStatus').className = 'pill ' + (r?.status || '');
  }
  function visibleIndices() {
    const q = $('searchInput').value.toLowerCase().trim(); const filter = $('statusFilter').value;
    return cases.map((c, i) => ({c, i})).filter(({c}) => {
      if(c.batch!==batch)return false;
      const s = db.records[c.candidate_id]?.status || 'pending';
      return (filter === 'all' || (filter === 'priority' ? c.ai_suggestion?.recommended_human_check : s === filter)) && (!q || [c.candidate_id, c.task_group_id, c.model_input.original_user_request, c.model_input.proposed_action.tool_name].join(' ').toLowerCase().includes(q));
    }).map(x => x.i);
  }
  function renderSidebar() {
    const indices = visibleIndices(), list = $('caseList'); list.replaceChildren();
    $('listCount').textContent = '显示 ' + indices.length + ' 条';
    for (const i of indices) {
      const c = cases[i], r = db.records[c.candidate_id];
      const button = document.createElement('button'); button.type = 'button'; button.className = 'case-item' + (i === current ? ' active' : '');
      button.setAttribute('aria-current', i === current ? 'true' : 'false');
      const number = document.createElement('span'); number.className = 'case-index'; number.textContent = String(i+1).padStart(3,'0');
      const text = document.createElement('span'); const title = document.createElement('strong'); title.textContent = c.model_input.proposed_action.tool_name;
      const task = document.createElement('small'); task.textContent = c.task_group_id.replace('workspace:', '') + ' · ' + statusText(r);
      text.append(title, task); const dot = document.createElement('span'); dot.className = 'status-dot ' + (r?.status === 'pending' ? 'draft' : r?.status || '');
      button.append(number,text,dot); button.addEventListener('click', () => go(i)); list.append(button);
    }
    if (!indices.length) { const empty = document.createElement('p'); empty.className = 'empty-list'; empty.textContent = '没有符合筛选条件的样本'; list.append(empty); }
  }
  function render() {
    const c = cases[current], m = c.model_input, r = record();
    $('caseNumber').textContent = '样本 ' + String(current+1).padStart(3,'0') + ' / ' + cases.length;
    $('userRequest').textContent = m.original_user_request;
    $('toolName').textContent = m.proposed_action.tool_name;
    $('actionArguments').textContent = Object.entries(m.proposed_action.arguments).map(([k,v]) => k + ':\n' + (typeof v === 'string' ? v : JSON.stringify(v,null,2))).join('\n\n');
    $('candidateId').textContent = c.candidate_id; $('taskGroup').textContent = c.task_group_id;
    $('observationCount').textContent = m.observations_before_call.length + ' 条完整记录';
    const observations = $('observations'); observations.replaceChildren();
    m.observations_before_call.forEach((o,i) => {
      const detail = document.createElement('details'); detail.className = 'observation'; detail.open = i === m.observations_before_call.length-1;
      const summary = document.createElement('summary'); summary.textContent = String(i+1).padStart(2,'0') + ' · ';
      const name = document.createElement('strong'); name.textContent = o.tool_name || '来源工具未记录'; summary.append(name);
      const args = document.createElement('div'); args.className = 'producing-args'; args.textContent = '来源参数：' + JSON.stringify(o.tool_arguments);
      const view = c.observation_views?.[i];
      const content = document.createElement('pre'); content.className = 'code-block'; content.tabIndex=0; content.textContent = view || o.content || '（空观察）';
      detail.append(summary,args);
      if(view && view !== o.content) {
        const controls = document.createElement('div'); controls.className='observation-view-controls';
        const hint = document.createElement('span'); hint.textContent='易读排版 · 仅展开原数据的格式转义';
        const toggle = document.createElement('button'); toggle.type='button'; toggle.className='text-button'; toggle.textContent='查看原文';
        let raw=false; toggle.onclick=()=>{raw=!raw;content.textContent=raw?o.content:view;toggle.textContent=raw?'恢复易读排版':'查看原文';hint.textContent=raw?'原始工具返回':'易读排版 · 仅展开原数据的格式转义';};
        controls.append(hint,toggle);detail.append(controls);
      }
      detail.append(content);
      if (o.error) { const error = document.createElement('div'); error.className='producing-args'; error.textContent='工具错误：'+o.error; detail.append(error); }
      observations.append(detail);
    });
    if (!m.observations_before_call.length) { const empty = document.createElement('p'); empty.className='muted'; empty.textContent='该调用前没有工具观察记录。'; observations.append(empty); }
    document.querySelectorAll('input[name="decision"]').forEach(el => el.checked = r?.decision === el.value);
    const suggestion=c.ai_suggestion;
    $('aiSuggestion').hidden=!suggestion;
    if(suggestion){$('aiDecision').textContent=suggestion.suggested_label===null?'建议：不确定，需人工确认':suggestion.suggested_label===0?'建议：授权（0）':'建议：未授权（1）';$('aiReason').textContent=suggestion.reason;$('aiEvidence').textContent=suggestion.evidence.map(e=>e.location+'\n'+e.excerpt).join('\n\n');db.assistanceViews ||= {};db.assistanceViews[c.candidate_id]=true;persist();}
    $('formError').hidden=true; const visible=visibleIndices();const pos=visible.indexOf(current);$('previousButton').disabled=pos<=0;$('nextButton').disabled=pos<0||pos===visible.length-1;
    updateDecisionUI(); updateProgress(); updateSaveStatus(); renderSidebar();
  }
  function updateDecisionUI() {
    const uncertain = decision()==='uncertain';
    $('confirmButton').firstChild.textContent = uncertain ? '标记待讨论并下一条 ' : '确认复核并下一条 ';
  }
  function draft() {
    if(!reviewer){openProfile();return;}
    const d = decision();
    if (!d && !record()) return;
    db.records[cases[current].candidate_id] = {...record(),decision:d,label:null,status:'pending',reason:record()?.reason||'',evidence:record()?.evidence||'',review_mode:'label_only',reviewer,ai_assisted:!!db.assistanceViews?.[cases[current].candidate_id],updated_at:new Date().toISOString()};
    $('formError').hidden=true; updateDecisionUI(); persist(); updateProgress(); renderSidebar();
  }
  function go(i) { current=Math.max(0,Math.min(cases.length-1,i)); persist(); render(); }
  function confirmReview() {
    if(!reviewer){openProfile();return;}
    const d=decision();
    const error=!d?'请先选择授权、未授权或不确定。':'';
    if (error) { $('formError').textContent=error; $('formError').hidden=false; return; }
    const visible=visibleIndices(), next=visible.find(i=>i>current);
    db.records[cases[current].candidate_id]={...record(),decision:d,label:d==='uncertain'?null:Number(d),status:d==='uncertain'?'disputed':'reviewed',reason:record()?.reason||'',evidence:record()?.evidence||'',review_mode:'label_only',reviewer,ai_assisted:!!db.assistanceViews?.[cases[current].candidate_id],updated_at:new Date().toISOString()};
    persist();
    if (next!==undefined) go(next); else { render(); toast('当前筛选范围已到最后一条。可以导出备份，或继续查看其他样本。'); }
  }
  function download(name, content, type) {
    const blob = new Blob([content],{type}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=name; document.body.append(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),30000); toast('已生成下载文件，请查看浏览器下载记录。');
  }
  function exportName(suffix) { return 'workspace400-'+reviewer.replace(/[^\p{L}\p{N}_-]/gu,'_')+'-'+new Date().toISOString().slice(0,10)+suffix; }
  function backupObject() { return {schema_version:2,dataset_id:data.dataset_id,dataset_sha256:data.dataset_sha256,reviewer,exported_at:new Date().toISOString(),cursor:current,batch,records:db.records,assistance_views:db.assistanceViews||{}}; }
  function labeledRows() {
    return cases.filter(c => {const r=db.records[c.candidate_id];return r?.status==='reviewed'&&(r.label===0||r.label===1)&&String(r.label)===r.decision&&r.reviewer===reviewer;}).map(c=>{const r=db.records[c.candidate_id];return {candidate_id:c.candidate_id,label:r.label,model_input:c.model_input,task_group_id:c.task_group_id,permitted_use:'train_development_only_after_review',annotation:{review_status:'reviewed',review_mode:r.review_mode||'detailed',reviewer:r.reviewer,reason:r.reason||'',evidence:r.evidence||'',ai_assisted:!!r.ai_assisted||!!db.assistanceViews?.[c.candidate_id],updated_at:r.updated_at}};});
  }
  function csvCell(value) { let s=String(value??'');if (/^[\s]*[=+@-]/.test(s)) s="'"+s;return '"'+s.replaceAll('"','""')+'"'; }
  function exportCsv() {
    const header=['row_no','candidate_id','label','review_status','reviewer','reason','evidence','task_group_id','ai_assisted','review_mode'];
    const rows=cases.map((c,i)=>{const r=db.records[c.candidate_id];return [i+1,c.candidate_id,r?.status==='reviewed'?r.label:'',r?.status||'pending',r?.reviewer||'',r?.reason||'',r?.evidence||'',c.task_group_id,!!r?.ai_assisted||!!db.assistanceViews?.[c.candidate_id],r?.review_mode||''];});
    return '\ufeff'+[header,...rows].map(r=>r.map(csvCell).join(',')).join('\r\n');
  }
  function validateBackup(b) {
    window.ReviewCore.validateBackup(b,data);
    if(b.reviewer!==reviewer)throw Error('这是 '+b.reviewer+' 的备份。请切换为该复核人；汇总他人结果请用“团队汇总”。');
    return b.records;
  }
  function move(offset){const visible=visibleIndices(),pos=visible.indexOf(current);if(pos>=0&&visible[pos+offset]!==undefined)go(visible[pos+offset]);}
  function applyFilters(){const visible=visibleIndices();if(visible.length&&!visible.includes(current))go(visible[0]);else render();}
  function openProfile(){$('profileInput').value=reviewer;$('profileBatch').value=String(batch);$('profileDialog').showModal();}
  $('previousButton').onclick=()=>move(-1); $('nextButton').onclick=()=>move(1);
  $('nextPending').onclick=()=>{const group=batchIndices();let i=group.find(i=>i>current&&db.records[cases[i].candidate_id]?.status!=='reviewed');if(i===undefined)i=group.find(i=>db.records[cases[i].candidate_id]?.status!=='reviewed');if(i===undefined)toast('本组已全部确认，请导出备份。');else{$('searchInput').value='';$('statusFilter').value='all';go(i);}};
  $('searchInput').oninput=applyFilters; $('statusFilter').onchange=applyFilters;
  $('batchSelect').onchange=()=>{batch=Number($('batchSelect').value);$('searchInput').value='';$('statusFilter').value='all';go(batchIndices()[0]);};
  $('shareBatch').onclick=async()=>{const url=new URL(window.location.href);url.search='';url.hash='';url.searchParams.set('batch',String(batch));try{await navigator.clipboard.writeText(url.href);toast('已复制本组链接。请组员填写自己的姓名代号。');}catch{window.prompt('复制这个分组链接：',url.href);}};
  document.querySelectorAll('input[name="decision"]').forEach(el=>el.onchange=draft);
  $('confirmButton').onclick=confirmReview;
  $('adoptSuggestion').onclick=()=>{const s=cases[current].ai_suggestion;if(!s)return;document.querySelector('input[name="decision"][value="'+(s.suggested_label===null?'uncertain':String(s.suggested_label))+'"]').checked=true;draft();toast('已选中 AI 建议。请核对后点击确认。');};
  $('expandAll').onclick=()=>document.querySelectorAll('.observation').forEach(d=>d.open=true);
  $('collapseAll').onclick=()=>document.querySelectorAll('.observation').forEach(d=>d.open=false);
  $('helpButton').onclick=$('guideLink').onclick=()=>$('helpDialog').showModal();
  document.querySelectorAll('.dialog-close').forEach(b=>b.onclick=()=>b.closest('dialog').close());
  $('exportButton').onclick=()=>{if(!reviewer){openProfile();return;}const all=Object.values(db.records), disputed=all.filter(r=>r.status==='disputed').length;$('exportSummary').textContent='复核人 '+reviewer+' · 已确认 '+labeledRows().length+' 条 · 待讨论 '+disputed+' 条。';$('exportDialog').showModal();};
  $('backupExport').onclick=()=>download(exportName('-backup.json'),JSON.stringify(backupObject(),null,2),'application/json');
  $('labelsExport').onclick=()=>{const rows=labeledRows();if(!rows.length){toast('还没有可导出的已确认标签。请选择标签并点击确认。');return;}download(exportName('-reviewed.jsonl'),rows.map(r=>JSON.stringify(r)).join('\n')+'\n','application/x-ndjson');};
  $('csvExport').onclick=()=>download(exportName('-review.csv'),exportCsv(),'text/csv;charset=utf-8');
  $('profileButton').onclick=openProfile;
  $('profileConfirm').onclick=()=>{const value=$('profileInput').value.trim();if(!value){toast('请填写复核人代号。');return;}persist();reviewer=value;try{localStorage.setItem('dsa4266-review400-reviewer',reviewer);}catch{}const targetBatch=Number($('profileBatch').value);$('profileDialog').close();loadProfile();batch=validBatch(targetBatch)?targetBatch:1;$('batchSelect').value=String(batch);$('searchInput').value='';$('statusFilter').value='all';go(cases[current].batch===batch?current:batchIndices()[0]);toast('已切换为 '+reviewer+'，显示此人的独立进度。');};
  $('importButton').onclick=()=>$('importFile').click();
  $('importFile').onchange=async event=>{
    const file=event.target.files[0];if(!file)return;
    try {if(file.size>20000000)throw Error('备份文件过大，请检查文件。');const b=JSON.parse(await file.text()), records=validateBackup(b);const count=Object.keys(records).length, conflicts=Object.keys(records).filter(id=>db.records[id]).length;
      if(!window.confirm('导入 '+count+' 条记录。'+(conflicts?'其中 '+conflicts+' 条在本机已有记录；继续将覆盖这些同 ID 记录。':'本机已有的其他记录会保留。')+'建议先导出当前备份。'))return;
      db.records={...db.records,...records};db.assistanceViews={...db.assistanceViews,...b.assistance_views};if(Number.isInteger(b.cursor))current=Math.max(0,Math.min(cases.length-1,b.cursor));batch=cases[current].batch;$('batchSelect').value=String(batch);$('statusFilter').value='all';$('searchInput').value='';persist();render();toast('备份已导入。');
    } catch(error){toast(error.message||'无法读取该备份。');}finally{event.target.value='';}
  };
  document.addEventListener('keydown',e=>{if(document.querySelector('dialog[open]'))return;if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();confirmReview();return;}if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;const d={'1':'0','2':'1','3':'uncertain'}[e.key];if(d){e.preventDefault();document.querySelector('input[name="decision"][value="'+d+'"]').checked=true;draft();}else if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}else if(e.key==='ArrowRight'){e.preventDefault();move(1);}});
  // Small pure hooks for deterministic local tests; not a network or model API.
  window.ReviewApp={backupObject,labeledRows,exportCsv,validateBackup,datasetSha:data.dataset_sha256};
  let teamBackups=[], teamResult=null;
  $('teamButton').onclick=()=>$('teamDialog').showModal();
  $('teamImport').onclick=()=>$('teamImportFile').click();
  $('teamImportFile').onchange=async event=>{
    const files=[...event.target.files];if(!files.length)return;
    try{
      if(files.length>20||files.some(f=>f.size>20000000))throw Error('每次最多导入 20 个备份，每份不超过 20 MB。');
      const incoming=await Promise.all(files.map(async f=>JSON.parse(await f.text())));
      const result=window.ReviewCore.mergeBackups([...teamBackups,...incoming],data);
      teamBackups=result.profiles;teamResult=result;
      $('teamFiles').textContent='已载入：'+result.profiles.map(b=>b.reviewer+'（'+Object.keys(b.records).length+' 条记录）').join('、');
      const confirmed=result.rows.filter(r=>r.status==='reviewed').length, conflicts=result.rows.filter(r=>r.status==='disputed');
      $('teamSummary').textContent='无分歧已确认 '+confirmed+' / 400 · 分歧或不确定 '+conflicts.length+' · 尚未确认 '+(400-confirmed-conflicts.length);
      const list=$('teamConflicts');list.replaceChildren();
      for(const r of conflicts){const i=cases.findIndex(c=>c.candidate_id===r.candidate_id);const box=document.createElement('div');box.className='conflict-row';const link=document.createElement('button');link.className='text-button';link.textContent='查看样本 '+String(i+1).padStart(3,'0');link.onclick=()=>{$('teamDialog').close();batch=cases[i].batch;$('batchSelect').value=String(batch);$('searchInput').value='';$('statusFilter').value='all';go(i);};const text=document.createElement('div');text.textContent=r.reviews.map(v=>v.reviewer+'：'+(v.status==='reviewed'?(v.label===0?'授权':'未授权'):v.status==='disputed'?'不确定':'草稿')).join('；');box.append(link,text);list.append(box);}
      $('teamLabels').disabled=confirmed===0;$('teamReport').disabled=false;
      toast('团队备份已汇总，个人进度未被覆盖。');
    }catch(error){toast(error.message||'团队备份无法读取。');}finally{event.target.value='';}
  };
  $('teamLabels').onclick=()=>{if(!teamResult)return;const rows=window.ReviewCore.mergedLabels(teamResult,data);if(!rows.length)return;download('workspace400-team-reviewed.jsonl',rows.map(r=>JSON.stringify(r)).join('\n')+'\n','application/x-ndjson');};
  $('teamReport').onclick=()=>{if(!teamResult)return;download('workspace400-team-report.json',JSON.stringify({dataset_id:data.dataset_id,dataset_sha256:data.dataset_sha256,exported_at:new Date().toISOString(),...teamResult},null,2),'application/json');};
  loadProfile();
  if(!reviewer)openProfile();
})();
