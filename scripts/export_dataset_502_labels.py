#!/usr/bin/env python3
"""Re-export 502 provisional labels after editing the 16-column annotation_sheet.csv.

Never adds reviewer metadata to model_input. Supports data already on current branch,
or the upstream origin/main branch (git fetch origin main first).
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess

SRC_REPO_PATH='data/delivery/dataset-502-v1/annotation_candidates.jsonl'
LABEL_DIR=Path('data/annotations/dataset_502_v1')
RESULT_PATH=Path('results/dataset_502_labeling_statistics.json')
COLUMNS=['row_no','call_id','episode_id','user_task_id','scenario_type','injection_task_id','original_user_request','tool_name','tool_arguments','observation_preview','label','injection_present','reason','evidence','review_status','reviewer']


def source_records(path=None):
    if path:
        text=Path(path).read_text(encoding='utf-8')
    elif Path(SRC_REPO_PATH).is_file():
        text=Path(SRC_REPO_PATH).read_text(encoding='utf-8')
    else:
        p=subprocess.run(['git','show',f'origin/main:{SRC_REPO_PATH}'],text=True,capture_output=True)
        if p.returncode: raise SystemExit('Cannot load source from origin/main. Run: git fetch origin main\n'+p.stderr)
        text=p.stdout
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def save_csv(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--source',help='Optional explicit annotation_candidates.jsonl path')
    arg.add_argument('--check-only',action='store_true',help='Validate without rewriting exported files')
    opt=arg.parse_args()
    src=source_records(opt.source)
    with (LABEL_DIR/'annotation_sheet.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=COLUMNS:raise ValueError('CSV columns mismatch; expected exactly original 16-column order')
        rows=list(reader)
    if len(rows)!=len(src)!=502:raise ValueError(f'Expected 502 source and CSV rows: {len(src)},{len(rows)}')
    extra_audit={}
    ap=LABEL_DIR/'annotation_audit.csv'
    if ap.exists():
        with ap.open(encoding='utf-8-sig',newline='') as f:
            extra_audit={int(r['row_no']):r for r in csv.DictReader(f)}
    entries=[]; group_count=Counter(); label_count=Counter(); inject_count=Counter()
    new_audit=[]
    for idx,(raw,row) in enumerate(zip(src,rows),1):
        act=raw['model_input']['proposed_action']; parent=(raw.get('parent_sources') or [{}])[0]
        expected_id=parent.get('call_id') or raw['candidate_id'] if raw['source_type']=='observed' else raw['candidate_id']
        expected_scenario=('constructed' if raw['source_type']=='constructed' else ('attack' if '/attack/' in parent.get('episode_id','') else 'normal'))
        if int(row['row_no'])!=idx or row['call_id']!=expected_id:raise ValueError(f'row {idx}: row_no/call_id changed')
        if row['tool_name']!=act['tool_name']:raise ValueError(f'row {idx}: tool_name changed')
        if json.loads(row['tool_arguments'])!=act['arguments']:raise ValueError(f'row {idx}: tool_arguments changed')
        if row['scenario_type']!=expected_scenario:raise ValueError(f'row {idx}: scenario_type changed')
        if row['original_user_request']!=raw['model_input']['original_user_request']:raise ValueError(f'row {idx}: original_user_request changed')
        label=row['label'].strip();ip=row['injection_present'].strip().upper()
        if label not in ('0','1') or ip not in ('TRUE','FALSE'):raise ValueError(f'row {idx}: missing/invalid label or injection_present')
        if not row['reason'].strip() or not row['evidence'].strip():raise ValueError(f'row {idx}: missing reason/evidence')
        status=row['review_status'].strip().lower()
        if status not in ('pending','reviewed','disputed'):raise ValueError(f'row {idx}: review_status must be pending/reviewed/disputed')
        if status=='reviewed' and not row['reviewer'].strip():raise ValueError(f'row {idx}: reviewed requires named reviewer')
        previous=extra_audit.get(idx,{})
        # Verify source fingerprint identity; prevents joining by row number to the wrong source revision.
        if previous and previous.get('model_input_fingerprint')!=raw['model_input_fingerprint']:
            raise ValueError(f'row {idx}: source model-input fingerprint differs from original audit')
        new=dict(raw)
        new['label']=int(label)
        new['injection_present']=(ip=='TRUE')
        new['annotation']={
            'reason':row['reason'].strip(), 'evidence':row['evidence'].strip(),
            'review_status':status, 'reviewer':row['reviewer'].strip(),
            'annotation_origin':previous.get('annotation_origin','CSV-imported annotation'),
            'review_priority':previous.get('review_priority','Standard'),
            'review_flags':previous.get('review_flags','').split(' | ') if previous.get('review_flags') else []
        }
        if set(new['model_input'])!={'original_user_request','observations_before_call','proposed_action'}:
            raise ValueError(f'row {idx}: source has unexpected model_input fields')
        entries.append(new)
        label_count[label]+=1; inject_count[ip]+=1;group_count[row['scenario_type']]+=1
        if previous:
            previous=dict(previous);previous.update({'label':label,'injection_present':ip,'review_status':status,'reviewer':row['reviewer'].strip()})
            new_audit.append(previous)
    if opt.check_only:
        # Also ensure exported JSONL matches the current CSV, not just source and CSV internal structure.
        existing=LABEL_DIR/'labeled_calls.jsonl'
        if not existing.exists():raise ValueError(f'Missing exported JSONL: {existing}')
        with existing.open(encoding='utf-8') as f:rows_out=[json.loads(x) for x in f if x.strip()]
        if rows_out!=entries:raise ValueError('JSONL is out of sync with CSV. Run exporter without --check-only')
        print('PASS: 502 rows, correct IDs, label fields, preserved model_input, matching JSONL')
        print('Label counts:',dict(label_count),'source:',dict(group_count),'injection:',dict(inject_count))
        return
    LABEL_DIR.mkdir(parents=True,exist_ok=True)
    with (LABEL_DIR/'labeled_calls.jsonl').open('w',encoding='utf-8') as f:
        for entry in entries:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    observed=[dict(r,row_no=i) for i,r in enumerate((x for x in rows if x['scenario_type']!='constructed'),1)]
    constructed=[dict(r,row_no=i) for i,r in enumerate((x for x in rows if x['scenario_type']=='constructed'),1)]
    save_csv(LABEL_DIR/'observed_382_annotation_sheet.csv',observed,COLUMNS)
    save_csv(LABEL_DIR/'constructed_120_annotation_sheet.csv',constructed,COLUMNS)
    if new_audit:
        save_csv(ap,new_audit,list(new_audit[0]))
    queue_file=LABEL_DIR/'review_queue.csv'
    if queue_file.exists():
        with queue_file.open(encoding='utf-8-sig',newline='') as f:
            queue_reader=csv.DictReader(f)
            queue_fields=queue_reader.fieldnames
            queue_rows=list(queue_reader)
        for queued in queue_rows:
            rno=int(queued['row_no'])
            if rno<1 or rno>502:raise ValueError(f'Bad review queue row number {rno}')
            queued['provisional_label']=rows[rno-1]['label']
        save_csv(queue_file,queue_rows,queue_fields)
    st={}
    if RESULT_PATH.exists():st=json.loads(RESULT_PATH.read_text(encoding='utf-8'))
    if not st:st={'dataset':'dataset-502-v1','counts':{}}
    st['counts'].update({
        'total_candidates':502,'observed':len(observed),'constructed':len(constructed),
        'label_0_provisional':label_count['0'],'label_1_provisional':label_count['1'],
        'injection_present_true':inject_count['TRUE'],'injection_present_false':inject_count['FALSE'],
        'review_pending':sum(x['annotation']['review_status']=='pending' for x in entries),
        'human_reviewed':sum(x['annotation']['review_status']=='reviewed' for x in entries),
        'review_disputed':sum(x['annotation']['review_status']=='disputed' for x in entries),
    })
    st['label_distribution_by_source']={t:dict(Counter(str(x['label']) for x in entries if x['source_type']==t)) for t in ('observed','constructed')}
    RESULT_PATH.parent.mkdir(parents=True,exist_ok=True)
    RESULT_PATH.write_text(json.dumps(st,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('EXPORT PASS: 502 rows, labels',dict(label_count),'injection',dict(inject_count))

if __name__=='__main__':main()
