import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
import {toCreasedNormals} from 'three/addons/utils/BufferGeometryUtils.js';

const $=id=>document.getElementById(id);
const api=async(url,opts={})=>{const r=await fetch(url,opts);let d;if((r.headers.get('content-type')||'').includes('application/json'))d=await r.json();else if(!r.ok)throw new Error(await r.text());else return r;if(!r.ok||d?.ok===false)throw new Error(d?.error||`Request failed (${r.status})`);return d;};
const post=(url,data)=>api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));

// ---------------------------------------------------------------- 3D scene
const scene=new THREE.Scene();scene.background=new THREE.Color('#111821');
const camera=new THREE.PerspectiveCamera(45,1,.1,100000);camera.up.set(0,0,1);camera.position.set(180,-220,180);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('canvas').appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,0,40);
scene.add(new THREE.HemisphereLight(0xffffff,0x47515c,2));
const key=new THREE.DirectionalLight(0xffffff,2.2);key.position.set(150,-180,260);scene.add(key);
const fill=new THREE.DirectionalLight(0x9bcfff,.8);fill.position.set(-180,100,120);scene.add(fill);
const grid=new THREE.GridHelper(500,50,0x667788,0x354250);grid.rotation.x=Math.PI/2;grid.position.z=0;scene.add(grid);
scene.add(new THREE.AxesHelper(70));

const loader=new STLLoader();
let sourceMesh=null,partMeshes=[],partData=[],modelData=[],activeModel=0,selectedPart=0,sourceVisible=true,explode=0,sourceMaxSize=100,generated=false,flatMode=false,generating=false,hasSource=false,lastStatus=null;
let sourceToken=0,partsToken=0;
const NOTE_3D='Includes the silicone gap and pour channel without double-counting their overlap.',NOTE_2D='Silicone fills the contoured well up to the top of the wall, minus the volume the original STL displaces.';
const PART_COLORS=['#e34b91','#9b58e8','#d5ae45','#73c95d','#39b8a9','#8793a8','#e07848','#6b9fe0'];
const IDENTITY_TF={scale:[1,1,1],rotate:[0,0,0],move:[0,0,0]};

function mat(color,rough=.7){return new THREE.MeshStandardMaterial({color,roughness:rough,metalness:.05,side:THREE.DoubleSide});}
function clearParts(){for(const m of partMeshes){scene.remove(m);m.geometry.dispose();}partMeshes=[];partData=[];}

// Creased normals: smooth shading on curved walls, crisp edges on flanges and caps.
function addSTL(url,material){return new Promise((resolve,reject)=>loader.load(url,g=>{
  let geo=g;
  try{if(g.attributes.position.count<1200000)geo=toCreasedNormals(g,Math.PI/5);else g.computeVertexNormals();}
  catch(e){g.computeVertexNormals();geo=g;}
  if(geo!==g)g.dispose();
  resolve(new THREE.Mesh(geo,material));
},undefined,reject));}

function sceneBox(){const box=new THREE.Box3();if(sourceMesh)box.expandByObject(sourceMesh);partMeshes.forEach(m=>{if(m.visible)box.expandByObject(m)});return box;}
function updateInfo(){const box=sceneBox();if(box.isEmpty()){$('sceneInfo').textContent='';return;}const size=box.getSize(new THREE.Vector3()),f=v=>v.toFixed(1);$('sceneInfo').innerHTML=`<span style="color:#ff5c5c;font-weight:600">X</span> ${f(size.x)} × <span style="color:#5ce05c;font-weight:600">Y</span> ${f(size.y)} × <span style="color:#5c9bff;font-weight:600">Z</span> ${f(size.z)} mm`;}
// Only called on upload / generate / project open, so editing a transform no longer re-frames the camera.
function fitCamera(){const box=sceneBox();if(box.isEmpty())return;const size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());const max=Math.max(size.x,size.y,size.z,10);controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(max*1.5,-max*1.8,max*1.3));camera.near=Math.max(.01,max/10000);camera.far=max*100;camera.updateProjectionMatrix();controls.update();updateInfo();}
function resize(){const w=$('canvas').clientWidth,h=$('canvas').clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);}
new ResizeObserver(resize).observe($('canvas'));resize();
(function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);})();

function setView(v){const box=sceneBox();const c=box.isEmpty()?new THREE.Vector3():box.getCenter(new THREE.Vector3());const s=box.isEmpty()?100:Math.max(...box.getSize(new THREE.Vector3()).toArray(),50);const d=s*2.5;const offsets={top:[0,0,d],front:[0,-d,0],back:[0,d,0],left:[-d,0,0],right:[d,0,0],iso:[d*1.15,-d*1.5,d*1.1]};camera.position.copy(c).add(new THREE.Vector3(...offsets[v]));camera.up.set(0,0,1);if(v==='top')camera.up.set(0,1,0);controls.target.copy(c);controls.update();document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view===v));}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.tabpage').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab').forEach(t=>t.setAttribute('aria-selected','false'));b.classList.add('active');b.setAttribute('aria-selected','true');$('tab-'+b.dataset.tab).classList.add('active');});

// ---------------------------------------------------------------- messages
function status(t,kind=''){const el=$('status');el.textContent=t;el.dataset.kind=kind;}
function toast(title,msg,kind='info'){const t=document.createElement('div');t.className='toast '+kind;const b=document.createElement('b');b.textContent=title;const p=document.createElement('span');p.textContent=msg;const x=document.createElement('button');x.type='button';x.textContent='Dismiss';x.onclick=()=>t.remove();t.append(b,p,x);$('toasts').append(t);if(kind==='info')setTimeout(()=>t.remove(),5000);}
function fmt(v){return Number(v||0).toFixed(2);}

// ---------------------------------------------------------------- settings
// Mould settings persist in localStorage and travel inside project files.
const SETTING_FIELDS=['resin_density','silicone_density','style','panels','margin','top_extra','wall','base','clearance','fill_diameter','fill_depth_percent','rim_width','rim_thickness'];
const SETTING_CHECKS=['core_enabled','label_tab'];
const SETTINGS_KEY='madhat.settings.v1';
function getSettings(){const s={};SETTING_FIELDS.forEach(id=>s[id]=$(id).value);SETTING_CHECKS.forEach(id=>s[id]=$(id).checked);return s;}
function applySettings(s){if(!s||typeof s!=='object')return;SETTING_FIELDS.forEach(id=>{if(s[id]!==undefined&&s[id]!==null)$(id).value=s[id];});SETTING_CHECKS.forEach(id=>{if(typeof s[id]==='boolean')$(id).checked=s[id];});syncCore();}
function saveSettings(){try{localStorage.setItem(SETTINGS_KEY,JSON.stringify(getSettings()));}catch(e){}}
function restoreSettings(){try{applySettings(JSON.parse(localStorage.getItem(SETTINGS_KEY)||'null'));}catch(e){}}

// ------------------------------------------------------------------ status
async function refreshStatus(){
  const d=await api('/api/status');
  generated=!!d.generated;flatMode=!!(d.generated&&d.flat);modelData=d.models;hasSource=!!d.has_source;
  sourceMaxSize=d.source?Math.max(...d.source.extents):100;
  if(activeModel>=modelData.length)activeModel=Math.max(0,modelData.length-1);
  renderModels();
  const visible=d.models.filter(x=>x.visible).length;
  $('modelInfo').textContent=d.models.length?`${d.models.length} model${d.models.length===1?'':'s'} loaded, ${visible} visible`:'No model loaded yet.';
  $('empty').hidden=d.models.length>0;
  $('generate').disabled=!d.has_source||generating;
  ['saveProject','deleteAll','showAll','deleteSelected','duplicate'].forEach(id=>$(id).disabled=!d.models.length);
  lastStatus=d;renderStats(d);
  if(d.generated){
    await renderParts(d.parts);
  }else{
    $('partsList').textContent='Generate a mould to see its parts.';
    partsToken++;clearParts();$('selectedPart').innerHTML='<option value="0">—</option>';
  }
}

// Volumes are shown in mL with the weight in grams beside them (volume x density).
const density=id=>{const v=+$(id).value;return Number.isFinite(v)&&v>0?v:0;};
function showVol(id,ml,dens){
  const el=$(id),g=dens?` <small class="muted">· ${(ml*dens).toFixed(1)} g</small>`:'';
  el.innerHTML=`${fmt(ml)} mL${g}`;
}
function renderStats(d){
  const resin=density('resin_density'),sil=density('silicone_density'),dash='—';
  if(d.source){showVol('stlFill',d.source.volume_ml,resin);showVol('stlMix',d.source.volume_ml*1.1,resin);}
  else{$('stlFill').textContent=dash;$('stlMix').textContent=dash;}
  if(d.generated){
    const fl=d.flat;
    if(fl){
      $('flatDims').hidden=false;
      $('flatDims').textContent=`Cavity footprint ${fmt(fl.footprint_x_mm)} × ${fmt(fl.footprint_z_mm)} mm (${fmt(fl.footprint_area_mm2/100)} cm²), fill depth ${fmt(fl.fill_depth_mm)} mm`;
      $('gapL').textContent='Empty well';showVol('gap',fl.well_volume_mm3/1000,sil);
      $('channelL').textContent='Minus model displacement';showVol('channel',fl.stl_volume_mm3/1000,sil);
      $('siliconeNote').textContent=NOTE_2D;
    }else{
      $('flatDims').hidden=true;
      $('gapL').textContent='Gap around the model';showVol('gap',d.capacity.gap_ml,sil);
      $('channelL').textContent='Pour channel';showVol('channel',d.capacity.channel_ml,sil);
      $('siliconeNote').textContent=NOTE_3D;
    }
    showVol('capacity',d.capacity.total_ml,sil);
    showVol('mix',d.capacity.mix_ml,sil);
  }else{
    $('flatDims').hidden=true;$('siliconeNote').textContent=NOTE_3D;
    $('gapL').textContent='Gap around the model';$('channelL').textContent='Pour channel';
    ['gap','channel','capacity','mix'].forEach(id=>$(id).textContent=dash);
  }
}
// Editing a density updates the grams straight away, no server round trip.
['resin_density','silicone_density'].forEach(id=>$(id).addEventListener('input',()=>{if(lastStatus)renderStats(lastStatus);}));

function renderModels(){
  const box=$('modelsList');box.innerHTML='';
  if(!modelData.length){box.textContent='No STL models loaded.';return;}
  modelData.forEach(m=>{
    const row=document.createElement('div');row.className='listitem';
    const check=document.createElement('input');check.type='checkbox';check.checked=m.visible;check.setAttribute('aria-label',`Show ${m.name}`);
    check.onchange=async()=>{await post(`/api/model/${m.index}/visibility`,{visible:check.checked});await refreshStatus();await refreshSource();};
    const name=document.createElement('button');name.textContent=m.name;name.className='modelname'+(m.index===activeModel?' active':'');name.title=m.name;
    name.onclick=()=>{activeModel=m.index;renderModels();syncTransformFields();};
    row.append(check,name);box.append(row);
  });
}

async function refreshSource(){
  const token=++sourceToken;let m=null;
  try{m=await addSTL('/api/preview/source?'+Date.now(),mat('#b8c3cc'));}catch(e){}
  if(token!==sourceToken){if(m)m.geometry.dispose();return;}
  if(sourceMesh){scene.remove(sourceMesh);sourceMesh.geometry.dispose();sourceMesh=null;}
  if(m){
    if(flatMode){m.material.polygonOffset=true;m.material.polygonOffsetFactor=-1;m.material.polygonOffsetUnits=-1;}
    // The server already places the plug exactly on the base, so no client-side offset.
    m.position.z=0;
    m.updateMatrixWorld(true);m.visible=sourceVisible;scene.add(m);sourceMesh=m;
  }
  updateInfo();
}

async function renderParts(parts){
  const token=++partsToken;clearParts();partData=parts;
  const list=$('partsList');list.innerHTML='';$('selectedPart').innerHTML='';
  for(const p of parts){
    const m=await addSTL(`/api/preview/part/${p.index}?${Date.now()}`,mat(PART_COLORS[p.index%8]));
    if(token!==partsToken){m.geometry.dispose();return;}
    m.userData.basePosition=m.position.clone();scene.add(m);partMeshes.push(m);
    const row=document.createElement('div');row.className='listitem';
    const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;cb.setAttribute('aria-label',`Show ${p.name}`);cb.onchange=()=>m.visible=cb.checked;
    const sw=document.createElement('i');sw.className='swatch';sw.style.background=PART_COLORS[p.index%8];
    const label=document.createElement('span');label.textContent=p.name;
    row.append(cb,sw,label);
    if(p.size_mm){const sz=document.createElement('small');sz.className='muted';sz.style.marginLeft='auto';sz.textContent=p.size_mm.map(v=>v.toFixed(0)).join(' × ')+' mm';sz.title='Printed size: X × Y × Z';row.append(sz);}
    list.append(row);
    const opt=document.createElement('option');opt.value=p.index;opt.textContent=p.name;$('selectedPart').append(opt);
  }
  applyExplode();updateInfo();
}
function applyExplode(){partMeshes.forEach((m,i)=>{const dir=partData[i]?.direction||[0,0,0];const dist=(explode/100)*(8+Math.max(30,sourceMaxSize*.30))*1.25;m.position.set(dir[0]*dist,-dir[2]*dist,(dir[1]||0)*dist);});}

// ------------------------------------------------------------------ upload
async function uploadFiles(files){
  if(!files||!files.length)return;
  const fd=new FormData();[...files].forEach(f=>fd.append('files',f));
  try{
    status('Uploading…');
    await api('/api/upload',{method:'POST',body:fd});
    await refreshStatus();activeModel=Math.max(0,modelData.length-1);renderModels();syncTransformFields();
    await refreshSource();fitCamera();
    status('Model loaded.','ok');
  }catch(err){status('Upload failed. '+err.message,'error');toast('Upload failed',err.message,'error');}
}
$('stlFiles').onchange=async e=>{await uploadFiles(e.target.files);e.target.value='';};
$('emptyUpload').onclick=()=>$('stlFiles').click();
const vp=document.querySelector('.viewport');
['dragenter','dragover'].forEach(t=>vp.addEventListener(t,e=>{e.preventDefault();vp.classList.add('dragging');}));
vp.addEventListener('dragleave',e=>{if(!vp.contains(e.relatedTarget))vp.classList.remove('dragging');});
vp.addEventListener('drop',e=>{e.preventDefault();vp.classList.remove('dragging');uploadFiles(e.dataTransfer.files);});

// ---------------------------------------------------------------- generate
function setBusy(on){
  generating=on;
  $('generate').disabled=on||!hasSource;
  $('generate').textContent=on?'Generating…':'Generate mould';
  $('genProgress').hidden=!on;$('cancelGen').hidden=!on;$('cancelGen').disabled=false;
  if(!on)$('genProgress').value=0;
}
// Poll the background job until it finishes; shows live progress.
async function watchJob(){
  setBusy(true);
  try{
    for(;;){
      const j=await api('/api/job');
      const pct=Math.round((j.fraction||0)*100);
      $('genProgress').value=pct;
      if(j.state==='running'){status(`${j.message}\n${pct}% · ${Math.round(j.elapsed||0)} s`);await sleep(500);continue;}
      return j;
    }
  }finally{setBusy(false);}
}
async function finishJob(j){
  if(j.state==='done'){
    await refreshStatus();await refreshSource();fitCamera();
    const warnings=j.warnings||[];
    status('Mould generated.'+(warnings.length?'\n\n'+warnings.join('\n'):''),'ok');
    warnings.forEach(w=>toast('Check this',w,'warn'));
    document.querySelector('[data-tab="preview"]').click();
  }else if(j.state==='cancelled'){
    status(j.message||'Cancelled.');
  }else if(j.state==='error'){
    status('Generation failed. '+j.error,'error');toast('Generation failed',j.error,'error');
  }
}
$('generate').onclick=async()=>{
  const d={style:$('style').value,panels:+$('panels').value,top_extra:+$('top_extra').value,wall:+$('wall').value,base:+$('base').value,clearance:+$('clearance').value,fill_diameter:+$('fill_diameter').value,fill_depth_percent:+$('fill_depth_percent').value,rim_width:+$('rim_width').value,rim_thickness:+$('rim_thickness').value,margin:+$('margin').value,label_tab:$('label_tab').checked,core:$('core_enabled').checked&&$('style').value!=='flat'};
  saveSettings();
  try{
    await post('/api/generate',d);
    await finishJob(await watchJob());
  }catch(e){status('Generation failed. '+e.message,'error');toast('Generation failed',e.message,'error');}
};
$('cancelGen').onclick=async()=>{$('cancelGen').disabled=true;try{await post('/api/generate/cancel',{});}catch(e){}};

const syncCore=()=>{const flat=$('style').value==='flat';document.body.dataset.style=flat?'flat':'multi';$('core_enabled').disabled=flat;$('label_tab').disabled=!flat;if(flat)$('core_enabled').checked=false;};
$('style').onchange=syncCore;syncCore();
[...SETTING_FIELDS,...SETTING_CHECKS].forEach(id=>$(id).addEventListener('change',saveSettings));

// -------------------------------------------------------------- view/export
$('showSource').onchange=()=>{sourceVisible=$('showSource').checked;if(sourceMesh)sourceMesh.visible=sourceVisible;};
function syncExplodeButton(){$('explodeBtn').textContent=explode>0?'ASSEMBLE MOLD':'EXPLODE MOLD';}
$('explode').oninput=()=>{explode=+$('explode').value;$('explodeLabel').textContent=explode+'%';syncExplodeButton();applyExplode();};
$('explodeBtn').onclick=()=>{$('explode').value=explode>0?0:100;$('explode').dispatchEvent(new Event('input'));};
syncExplodeButton();
$('selectedPart').onchange=()=>selectedPart=+$('selectedPart').value;
$('exportSelected').onclick=()=>{if(partMeshes.length)location.href=`/api/export/part/${$('selectedPart').value}`;};
$('exportAll').onclick=()=>location.href='/api/export/zip';
$('exportExploded').onclick=()=>location.href='/api/export/zip?explode='+explode;
$('exportPlug').onclick=()=>location.href='/api/export/plug';

// ------------------------------------------------------------ model actions
$('deleteAll').onclick=async()=>{if(!confirm('Delete all imported STL models and generated parts?'))return;await api('/api/models',{method:'DELETE'});activeModel=0;await refreshStatus();await refreshSource();syncTransformFields();status('All models deleted.');};
$('showAll').onclick=async()=>{await post('/api/models/show-all',{});await refreshStatus();await refreshSource();};
$('deleteSelected').onclick=async()=>{if(!modelData.length)return;await api(`/api/model/${activeModel}`,{method:'DELETE'});await refreshStatus();await refreshSource();syncTransformFields();};
$('duplicate').onclick=async()=>{if(!modelData.length)return;await post(`/api/model/${activeModel}/duplicate`,{});await refreshStatus();activeModel=modelData.length-1;renderModels();syncTransformFields();await refreshSource();};

// ---------------------------------------------------------------- transform
const transformIds=['sx','sy','sz','rx','ry','rz','mx','my','mz'];let transformTimer=null;
// The fields always show the *selected model's* stored transform, so editing one
// value no longer resets the others.
function syncTransformFields(){
  const tf=modelData[activeModel]?.tf||IDENTITY_TF;
  ['x','y','z'].forEach((a,i)=>{$('s'+a).value=tf.scale[i];$('r'+a).value=tf.rotate[i];$('m'+a).value=tf.move[i];});
}
function transformPayload(){return {scale:[+$('sx').value,+$('sy').value,+$('sz').value],rotate:[+$('rx').value,+$('ry').value,+$('rz').value],move:[+$('mx').value,+$('my').value,+$('mz').value]};}
let transformBusy=false,transformPending=false;
async function applyTransform(){
  if(!modelData.length)return;
  if(transformBusy){transformPending=true;return;}
  transformBusy=true;
  try{
    do{
      transformPending=false;
      await post(`/api/model/${activeModel}/transform`,transformPayload());
      await refreshStatus();await refreshSource();
      $('transformStatus').textContent='Transform applied. Regenerate the mould to rebuild from this geometry.';$('transformStatus').dataset.kind='ok';
    }while(transformPending);
  }catch(e){$('transformStatus').textContent='Transform failed. '+e.message;$('transformStatus').dataset.kind='error';}
  finally{transformBusy=false;}
}
transformIds.forEach(id=>$(id).addEventListener('input',()=>{if($('uniform').checked&&['sx','sy','sz'].includes(id)){['sx','sy','sz'].forEach(k=>$(k).value=$(id).value);}clearTimeout(transformTimer);transformTimer=setTimeout(applyTransform,180);}));
$('resetTransform').onclick=()=>{transformIds.forEach(id=>$(id).value=(id[0]==='s'?1:0));applyTransform();};

// ------------------------------------------------------------------ project
function downloadBlob(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href);}
$('saveProject').onclick=async()=>{try{const r=await post('/api/project',{settings:getSettings()});downloadBlob(await r.blob(),'Mad_Hat_Project.zip');}catch(e){toast('Could not save project',e.message,'error');}};
$('openProject').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  const fd=new FormData();fd.append('file',file);
  try{
    const res=await api('/api/project/open',{method:'POST',body:fd});
    applySettings(res.settings);saveSettings();
    activeModel=0;await refreshStatus();syncTransformFields();await refreshSource();fitCamera();status('Project opened.','ok');
  }catch(err){status(err.message,'error');toast('Could not open project',err.message,'error');}
  e.target.value='';
};

// -------------------------------------------------------------------- start
restoreSettings();
(async()=>{
  try{
    await refreshStatus();syncTransformFields();await refreshSource();fitCamera();
    // If the page was reloaded while a mould was building, pick the job back up.
    const j=await api('/api/job');
    if(j.state==='running')await finishJob(await watchJob());
  }catch(e){status(e.message);}
})();
