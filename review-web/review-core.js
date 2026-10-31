(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ReviewCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  'use strict';
  function validateBackup(b, data) {
    if (!b || b.schema_version !== 2 || b.dataset_id !== data.dataset_id || b.dataset_sha256 !== data.dataset_sha256)
      throw Error('备份不属于这批 400 条数据。请勿导入旧版 120 条备份。');
    if (typeof b.reviewer !== 'string' || !b.reviewer.trim() || b.reviewer.length > 40)
      throw Error('复核人代号无效。');
    if (!Number.isFinite(Date.parse(b.exported_at))) throw Error('备份导出时间无效。');
    if (!b.records || typeof b.records !== 'object' || Array.isArray(b.records)) throw Error('备份记录格式无效。');
    const ids = new Set(data.cases.map(c => c.candidate_id));
    for (const [id,r] of Object.entries(b.records)) {
      if (!ids.has(id) || !r || r.reviewer !== b.reviewer || !['pending','reviewed','disputed'].includes(r.status) || ![null,'0','1','uncertain'].includes(r.decision))
        throw Error('备份包含不匹配的样本、复核人或状态。');
      if (!Number.isFinite(Date.parse(r.updated_at)) || (r.ai_assisted !== undefined && typeof r.ai_assisted !== 'boolean'))
        throw Error('备份时间或 AI 辅助标记无效。');
      if (r.status === 'reviewed' && (![0,1].includes(r.label) || String(r.label) !== r.decision))
        throw Error('已确认标签与选择不一致。');
      if (r.status !== 'reviewed' && r.label !== null) throw Error('草稿及待讨论记录不能带正式标签。');
      if (r.status === 'disputed' && r.decision !== 'uncertain') throw Error('待讨论记录必须标记不确定。');
    }
    return b;
  }
  function mergeBackups(backups, data) {
    const profiles = new Map();
    for (const backup of backups) {
      const b = validateBackup(backup, data), old = profiles.get(b.reviewer);
      if (old && Date.parse(old.exported_at) === Date.parse(b.exported_at) && JSON.stringify(old.records) !== JSON.stringify(b.records))
        throw Error('复核人 '+b.reviewer+' 有同一导出时间的不同备份，请只保留正确的一份。');
      if (!old || Date.parse(b.exported_at) > Date.parse(old.exported_at)) profiles.set(b.reviewer,b);
    }
    const rows = data.cases.map(c => {
      const reviews = [...profiles.values()].filter(b => b.records[c.candidate_id]).map(b => ({...b.records[c.candidate_id]}));
      const confirmed = reviews.filter(r => r.status === 'reviewed');
      const labels = new Set(confirmed.map(r => r.label));
      const conflict = labels.size > 1 || reviews.some(r => r.status === 'disputed');
      return {candidate_id:c.candidate_id, label:!conflict && labels.size === 1 ? [...labels][0] : null,
        status:conflict ? 'disputed' : labels.size === 1 ? 'reviewed' : 'pending',
        reviewer_count:confirmed.length, reviews};
    });
    return {profiles:[...profiles.values()], rows};
  }
  function mergedLabels(result, data) {
    const originals = new Map(data.cases.map(c => [c.candidate_id,c]));
    return result.rows.filter(r => r.status === 'reviewed').map(r => {
      const c=originals.get(r.candidate_id);
      return {candidate_id:r.candidate_id,label:r.label,model_input:c.model_input,task_group_id:c.task_group_id,
        permitted_use:'train_development_only_after_review',annotation:{review_status:'reviewed',review_mode:'label_only',
          aggregation:'agreement_of_submitted_confirmed_reviews',reviewer_count:r.reviewer_count,
          ai_assisted:r.reviews.some(v=>v.ai_assisted),reviews:r.reviews}};
    });
  }
  return {validateBackup,mergeBackups,mergedLabels};
});
