const test=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const path=require('node:path');
const root=path.resolve(__dirname,'..');
function app(storage=new Map([['dsa4266-review400-reviewer','tester']]),search=''){
  const elements=new Map(),radios=[];
  class Element{
    constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.value='';this.style={};this.textContent='';this.hidden=false;this.open=false;this.firstChild={textContent:''};this.files=[];}
    append(...nodes){this.children.push(...nodes);}
    replaceChildren(...nodes){this.children=[...nodes];}
    setAttribute(k,v){this[k]=v;}
    addEventListener(k,fn){this['on'+k]=fn;}
    remove(){}
    click(){this.onclick?.();}
    showModal(){this.open=true;}
    close(){this.open=false;}
    set checked(v){if(v)radios.forEach(r=>r._checked=false);this._checked=v;}
    get checked(){return !!this._checked;}
  }
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
  for(const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g)){const el=new Element(match[1]);el.id=match[2];elements.set(el.id,el);}
  for(const value of ['0','1','uncertain']){const el=new Element('input');el.value=value;radios.push(el);}
  elements.get('statusFilter').value='all';
  const document={body:new Element('body'),activeElement:new Element('body'),getElementById:id=>elements.get(id),createElement:tag=>new Element(tag),addEventListener(){},
    querySelectorAll:selector=>selector==='input[name="decision"]'?radios:[],
    querySelector:selector=>{if(selector==='input[name="decision"]:checked')return radios.find(r=>r.checked);if(selector==='dialog[open]')return [...elements.values()].find(e=>e.tagName==='DIALOG'&&e.open);const match=selector.match(/value="([^"]+)"/);return match?radios.find(r=>r.value===match[1]):null;}};
  const window={confirm:()=>true,location:{search,href:'http://localhost/'+search}};const context={window,document,URLSearchParams,navigator:{},localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,String(v))},setTimeout:()=>0,clearTimeout(){},Blob:class{},URL:{createObjectURL:()=>'',revokeObjectURL(){}},console};
  vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(root,'dataset.js'),'utf8'),context);window.ReviewCore=require('../review-core.js');vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),context);
  return {api:window.ReviewApp,data:window.REVIEW_DATA,elements,radios,storage,choose(value){const r=radios.find(r=>r.value===value);r.checked=true;r.onchange();},fill(id,value){const e=elements.get(id);e.value=value;e.oninput();},click(id){elements.get(id).onclick();}};
}
test('first use asks for identity; no reason/evidence inputs',()=>{
 const a=app(new Map());assert.equal(a.elements.get('profileDialog').open,true);assert.equal(a.elements.has('reasonInput'),false);assert.equal(a.elements.has('evidenceInput'),false);
 a.choose('0');a.click('confirmButton');assert.equal(a.api.labeledRows().length,0);
});
test('batch deep link, label-only confirmation, drafts, and group boundaries',()=>{
 const a=app(undefined,'?batch=2');assert.equal(a.api.backupObject().cursor,100);assert.equal(a.elements.get('caseList').children.length,100);
 a.choose('1');assert.equal(a.api.labeledRows().length,0);a.click('confirmButton');assert.equal(a.api.labeledRows().length,1);assert.equal(a.api.backupObject().cursor,101);
 assert.equal(a.api.labeledRows()[0].annotation.ai_assisted,false);
 for(let i=0;i<110;i++)a.click('nextButton');assert.equal(a.api.backupObject().cursor,199);
 a.choose('0');a.click('confirmButton');assert.equal(a.api.backupObject().cursor,199);
});
test('uncertain is saved without text and excluded from labeled export',()=>{
 const a=app();a.choose('uncertain');a.click('confirmButton');assert.equal(a.api.labeledRows().length,0);assert.equal(Object.values(a.api.backupObject().records)[0].status,'disputed');
});
test('reload preserves progress, reviewer isolation, edit invalidation',()=>{
 const a=app();a.choose('0');a.click('confirmButton');const b=app(a.storage);assert.equal(b.api.backupObject().cursor,1);assert.equal(b.api.labeledRows().length,1);
 b.click('previousButton');b.choose('1');assert.equal(b.api.labeledRows().length,0);
 b.elements.get('profileInput').value='other';b.elements.get('profileBatch').value='3';b.click('profileConfirm');assert.equal(b.api.backupObject().cursor,200);assert.equal(Object.keys(b.api.backupObject().records).length,0);
});
test('next pending stays within the assigned group and new backups validate',()=>{
 const a=app(undefined,'?batch=4');a.choose('0');a.click('confirmButton');a.click('nextPending');assert.equal(a.api.backupObject().cursor,302);
 assert.doesNotThrow(()=>a.api.validateBackup(a.api.backupObject()));assert.throws(()=>a.api.validateBackup({...a.api.backupObject(),reviewer:'other'}));
});
test('team file import combines four backups without changing individual progress and rejects bad files atomically',async()=>{
 const a=app();const template=a.api.backupObject(),time=new Date().toISOString();
 const files=[1,2,3,4].map(batch=>({size:100,text:async()=>JSON.stringify({...template,reviewer:'member'+batch,records:Object.fromEntries(a.data.cases.filter(c=>c.batch===batch).map(c=>[c.candidate_id,{reviewer:'member'+batch,label:0,decision:'0',status:'reviewed',updated_at:time,ai_assisted:false}]))})}));
 await a.elements.get('teamImportFile').onchange({target:{files,value:'files'}});
 assert.match(a.elements.get('teamSummary').textContent,/400 \/ 400/);assert.equal(a.elements.get('teamLabels').disabled,false);assert.equal(Object.keys(a.api.backupObject().records).length,0);
 await a.elements.get('teamImportFile').onchange({target:{files:[{size:10,text:async()=>'{"schema_version":1}'}],value:'bad'}});
 assert.match(a.elements.get('teamSummary').textContent,/400 \/ 400/);assert.match(a.elements.get('toast').textContent,/不属于/);
});
