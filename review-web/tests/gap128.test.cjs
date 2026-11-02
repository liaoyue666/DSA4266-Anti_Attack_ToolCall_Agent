const test=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const path=require('node:path');
const root=path.resolve(__dirname,'../gap128');
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
  vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(root,'dataset.js'),'utf8'),context);window.ReviewCore=require('../gap128/review-core.js');vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),context);
  return {api:window.ReviewApp,data:window.REVIEW_DATA,elements,radios,storage,choose(value){const r=radios.find(r=>r.value===value);r.checked=true;r.onchange();},fill(id,value){const e=elements.get(id);e.value=value;e.oninput();},click(id){elements.get(id).onclick();}};
}

module.exports={app};
