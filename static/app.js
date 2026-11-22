import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';

const $=id=>document.getElementById(id), api=async(url,opts={})=>{const r=await fetch(url,opts);let d;if((r.headers.get('content-type')||'').includes('application/json'))d=await r.json();else if(!r.ok)throw new Error(await r.text());else return r;if(!r.ok||d?.ok===false)throw new Error(d?.error||`Request failed (${r.status})`);return d;};
const post=(url,data)=>api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
const scene=new THREE.Scene();scene.background=new THREE.Color('#111821');
const camera=new THREE.PerspectiveCamera(45,1,.1,100000);camera.up.set(0,0,1);camera.position.set(180,-220,180);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('canvas').appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,0,40);
scene.add(new THREE.HemisphereLight(0xffffff,0x47515c,2));const key=new THREE.DirectionalLight(0xffffff,2.2);key.position.set(150,-180,260);scene.add(key);const fill=new THREE.DirectionalLight(0x9bcfff,.8);fill.position.set(-180,100,120);scene.add(fill);
const grid=new THREE.GridHelper(500,50,0x667788,0x354250);grid.rotation.x=Math.PI/2;grid.position.z=0;scene.add(grid);const axes=new THREE.AxesHelper(70);scene.add(axes);
const loader=new STLLoader();let sourceMesh=null,partMeshes=[],partData=[],modelData=[],activeModel=0,selectedPart=0,sourceVisible=true,explode=0,sourceMaxSize=100,generated=false,flatMode=false;
function mat(color,rough=.7){return new THREE.MeshStandardMaterial({color,roughness:rough,metalness:.05,side:THREE.DoubleSide});}
let sourceToken=0,partsToken=0;
const NOTE_3D='Includes the silicone gap and pour channel without double-counting their overlap.',NOTE_2D='Silicone fills the contoured well up to the top of the wall, minus the volume the original STL displaces.';
function clearParts(){for(const m of partMeshes)scene.remove(m);partMeshes=[];partData=[];}
function addSTL(url,material){return new Promise((resolve,reject)=>loader.load(url,g=>{g.computeVertexNormals();const m=new THREE.Mesh(g,material);m.castShadow=true;m.receiveShadow=true;resolve(m);},undefined,reject));}
function fitCamera(){const box=new THREE.Box3();if(sourceMesh)box.expandByObject(sourceMesh);partMeshes.forEach(m=>{if(m.visible)box.expandByObject(m)});if(box.isEmpty())return;const size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());const max=Math.max(size.x,size.y,size.z,10);controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(max*1.5,-max*1.8,max*1.3));camera.near=Math.max(.01,max/10000);camera.far=max*100;camera.updateProjectionMatrix();controls.update();const f=v=>v.toFixed(1);$('sceneInfo').innerHTML=`<span style="color:#ff5c5c;font-weight:600">X</span> ${f(size.x)} × <span style="color:#5ce05c;font-weight:600">Y</span> ${f(size.y)} × <span style="color:#5c9bff;font-weight:600">Z</span> ${f(size.z)} mm`;}
function resize(){const w=$('canvas').clientWidth,h=$('canvas').clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);}
new ResizeObserver(resize).observe($('canvas'));resize();
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}animate();
function setView(v){const box=new THREE.Box3();if(sourceMesh)box.expandByObject(sourceMesh);partMeshes.forEach(m=>{if(m.visible)box.expandByObject(m)});const c=box.isEmpty()?new THREE.Vector3():box.getCenter(new THREE.Vector3());const s=box.isEmpty()?100:Math.max(...box.getSize(new THREE.Vector3()).toArray(),50);const d=s*2.5;const offsets={top:[0,0,d],front:[0,-d,0],back:[0,d,0],left:[-d,0,0],right:[d,0,0],iso:[d*1.15,-d*1.5,d*1.1]};camera.position.copy(c).add(new THREE.Vector3(...offsets[v]));camera.up.set(0,0,1);if(v==='top')camera.up.set(0,1,0);controls.target.copy(c);controls.update();document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view===v));}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.tabpage').forEach(x=>x.classList.remove('active'));b.classList.add('active');$('tab-'+b.dataset.tab).classList.add('active');});
function status(t,kind=''){const el=$('status');el.textContent=t;el.dataset.kind=kind;}
function toast(title,msg,kind='info'){const t=document.createElement('div');t.className='toast '+kind;const b=document.createElement('b');b.textContent=title;const p=document.createElement('span');p.textContent=msg;const x=document.createElement('button');x.type='button';x.textContent='Dismiss';x.onclick=()=>t.remove();t.append(b,p,x);$('toasts').append(t);if(kind!=='error')setTimeout(()=>t.remove(),5000);}
const PART_COLORS=['#e34b91','#9b58e8','#d5ae45','#73c95d','#39b8a9','#8793a8','#e07848','#6b9fe0'];
function fmt(v){return Number(v||0).toFixed(2);}
async function refreshStatus(){
  const d=await api('/api/status');
  generated=!!d.generated;flatMode=!!(d.generated&&d.flat);modelData=d.models;
  sourceMaxSize=d.source?Math.max(...d.source.extents):100;
  renderModels();
  const visible=d.models.filter(x=>x.visible).length;
  $('modelInfo').textContent=d.models.length?`${d.models.length} model${d.models.length===1?'':'s'} loaded, ${visible} visible`:'No model loaded yet.';
  $('empty').hidden=d.models.length>0;
  $('generate').disabled=!d.has_source;
  ['saveProject','deleteAll','showAll','deleteSelected','duplicate'].forEach(id=>$(id).disabled=!d.models.length);
  const mL=v=>`${fmt(v)} mL`,dash='—';
  $('stlFill').textContent=d.source?mL(d.source.volume_ml):dash;
  $('stlMix').textContent=d.source?mL(d.source.volume_ml*1.1):dash;
  if(d.generated){
    const fl=d.flat;
    if(fl){
      $('flatDims').hidden=false;
      $('flatDims').textContent=`Cavity footprint ${fmt(fl.footprint_x_mm)} × ${fmt(fl.footprint_z_mm)} mm (${fmt(fl.footprint_area_mm2/100)} cm²), fill depth ${fmt(fl.fill_depth_mm)} mm`;
      $('gapL').textContent='Empty well';$('gap').textContent=mL(fl.well_volume_mm3/1000);
      $('channelL').textContent='Minus model displacement';$('channel').textContent=mL(fl.stl_volume_mm3/1000);
      $('siliconeNote').textContent=NOTE_2D;
    }else{
      $('flatDims').hidden=true;
      $('gapL').textContent='Gap around the model';$('gap').textContent=mL(d.capacity.gap_ml);
      $('channelL').textContent='Pour channel';$('channel').textContent=mL(d.capacity.channel_ml);
      $('siliconeNote').textContent=NOTE_3D;
    }
    $('capacity').textContent=mL(d.capacity.total_ml);
    $('mix').textContent=mL(d.capacity.mix_ml);
    renderParts(d.parts);
  }else{
    $('flatDims').hidden=true;$('siliconeNote').textContent=NOTE_3D;
    $('gapL').textContent='Gap around the model';$('channelL').textContent='Pour channel';
    ['gap','channel','capacity','mix'].forEach(id=>$(id).textContent=dash);
    $('partsList').textContent='Generate a mould to see its parts.';
    partsToken++;clearParts();$('selectedPart').innerHTML='<option value="0">—</option>';
  }
}
function renderModels(){const box=$('modelsList');box.innerHTML='';if(!modelData.length){box.textContent='No STL models loaded.';return;}modelData.forEach(m=>{const row=document.createElement('div');row.className='listitem';const check=document.createElement('input');check.type='checkbox';check.checked=m.visible;check.onchange=async()=>{await post(`/api/model/${m.index}/visibility`,{visible:check.checked});await refreshStatus();await refreshSource();};const name=document.createElement('button');name.textContent=m.name;name.className='modelname'+(m.index===activeModel?' active':'');name.title=m.name;name.onclick=()=>{activeModel=m.index;renderModels();loadTransformDefaults();};row.append(check,name);box.append(row);});}
async function refreshSource(){const token=++sourceToken;let m=null;try{m=await addSTL('/api/preview/source?'+Date.now(),mat('#b8c3cc'));}catch(e){}if(token!==sourceToken){if(m)m.geometry.dispose();return;}if(sourceMesh){scene.remove(sourceMesh);sourceMesh.geometry.dispose();sourceMesh=null;}if(m){if(flatMode){m.material.polygonOffset=true;m.material.polygonOffsetFactor=-1;m.material.polygonOffsetUnits=-1;}m.geometry.computeBoundingBox();if(generated&&!flatMode&&m.geometry.boundingBox){const actualBottom=m.geometry.boundingBox.min.z;const baseThickness=Math.max(0,Number($('base').value)||0);const actualBaseTop=Math.max(0,baseThickness-0.5);m.position.z=actualBaseTop-actualBottom+0.2;}else{m.position.z=0;}m.updateMatrixWorld(true);m.visible=sourceVisible;scene.add(m);sourceMesh=m;}fitCamera();}
async function renderParts(parts){const token=++partsToken;clearParts();partData=parts;const list=$('partsList');list.innerHTML='';$('selectedPart').innerHTML='';for(const p of parts){const m=await addSTL(`/api/preview/part/${p.index}?${Date.now()}`,mat(PART_COLORS[p.index%8]));if(token!==partsToken){m.geometry.dispose();return;}m.userData.basePosition=m.position.clone();scene.add(m);partMeshes.push(m);const row=document.createElement('div');row.className='listitem';const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;cb.onchange=()=>m.visible=cb.checked;const sw=document.createElement('i');sw.className='swatch';sw.style.background=PART_COLORS[p.index%8];const label=document.createElement('span');label.textContent=p.name;row.append(cb,sw,label);list.append(row);const opt=document.createElement('option');opt.value=p.index;opt.textContent=p.name;$('selectedPart').append(opt);}applyExplode();fitCamera();}
function applyExplode(){partMeshes.forEach((m,i)=>{const dir=partData[i]?.direction||[0,0,0];const dist=(explode/100)*(8+Math.max(30,sourceMaxSize*.30))*1.25;m.position.set(dir[0]*dist,-dir[2]*dist,(dir[1]||0)*dist);});}
async function uploadFiles(files){
  if(!files||!files.length)return;
  const fd=new FormData();[...files].forEach(f=>fd.append('files',f));
  try{
    status('Uploading…');
    await api('/api/upload',{method:'POST',body:fd});
    await refreshStatus();activeModel=Math.max(0,modelData.length-1);renderModels();await refreshSource();
    status('Model loaded.','ok');
  }catch(err){status('Upload failed. '+err.message,'error');toast('Upload failed',err.message,'error');}
}
$('stlFiles').onchange=async e=>{await uploadFiles(e.target.files);e.target.value='';};
$('emptyUpload').onclick=()=>$('stlFiles').click();
const vp=document.querySelector('.viewport');
['dragenter','dragover'].forEach(t=>vp.addEventListener(t,e=>{e.preventDefault();vp.classList.add('dragging');}));
vp.addEventListener('dragleave',e=>{if(!vp.contains(e.relatedTarget))vp.classList.remove('dragging');});
vp.addEventListener('drop',e=>{e.preventDefault();vp.classList.remove('dragging');uploadFiles(e.dataTransfer.files);});
$('generate').onclick=async()=>{const d={style:$('style').value,panels:+$('panels').value,top_extra:+$('top_extra').value,wall:+$('wall').value,base:+$('base').value,clearance:+$('clearance').value,fill_diameter:+$('fill_diameter').value,fill_depth_percent:+$('fill_depth_percent').value,rim_width:+$('rim_width').value,rim_thickness:+$('rim_thickness').value,margin:+$('margin').value,label_tab:$('label_tab').checked,core:$('core_enabled').checked&&$('style').value!=='flat'};$('generate').disabled=true;$('generate').textContent='Generating…';status('Building the mould. Large models can take a minute or two.');try{const res=await post('/api/generate',d);await refreshStatus();await refreshSource();status('Mould generated.'+(res.warnings&&res.warnings.length?'\n\n'+res.warnings.join('\n'):''),'ok');document.querySelector('[data-tab="preview"]').click();}catch(e){status('Generation failed. '+e.message,'error');toast('Generation failed',e.message,'error');}finally{$('generate').disabled=false;$('generate').textContent='Generate mould';}};
const syncCore=()=>{const flat=$('style').value==='flat';document.body.dataset.style=flat?'flat':'multi';$('core_enabled').disabled=flat;$('label_tab').disabled=!flat;if(flat)$('core_enabled').checked=false;};$('style').onchange=syncCore;syncCore();
$('showSource').onchange=()=>{sourceVisible=$('showSource').checked;if(sourceMesh)sourceMesh.visible=sourceVisible;};
function syncExplodeButton(){ $('explodeBtn').textContent=explode>0?'ASSEMBLE MOLD':'EXPLODE MOLD'; }
$('explode').oninput=()=>{explode=+$('explode').value;$('explodeLabel').textContent=explode+'%';syncExplodeButton();applyExplode();};
$('explodeBtn').onclick=()=>{$('explode').value=explode>0?0:100;$('explode').dispatchEvent(new Event('input'));};
syncExplodeButton();
$('selectedPart').onchange=()=>selectedPart=+$('selectedPart').value;
$('exportSelected').onclick=()=>{if(partMeshes.length)location.href=`/api/export/part/${$('selectedPart').value}`;};$('exportAll').onclick=()=>location.href='/api/export/zip';$('exportExploded').onclick=()=>location.href='/api/export/zip?explode='+explode;$('exportPlug').onclick=()=>location.href='/api/export/plug';
$('deleteAll').onclick=async()=>{if(!confirm('Delete all imported STL models and generated parts?'))return;for(let i=modelData.length-1;i>=0;i--)await api(`/api/model/${i}`,{method:'DELETE'});await refreshStatus();await refreshSource();status('All models deleted.');};
$('showAll').onclick=async()=>{await post('/api/models/show-all',{});await refreshStatus();await refreshSource();};$('deleteSelected').onclick=async()=>{if(!modelData.length)return;await api(`/api/model/${activeModel}`,{method:'DELETE'});activeModel=Math.max(0,Math.min(activeModel,modelData.length-2));await refreshStatus();await refreshSource();};$('duplicate').onclick=async()=>{if(modelData.length){await post(`/api/model/${activeModel}/duplicate`,{});activeModel=modelData.length;await refreshStatus();await refreshSource();}};
const transformIds=['sx','sy','sz','rx','ry','rz','mx','my','mz'];let transformTimer=null;
function loadTransformDefaults(){transformIds.forEach(id=>$(id).value=(id[0]==='s'?1:0));}
function transformPayload(){return {scale:[+$('sx').value,+$('sy').value,+$('sz').value],rotate:[+$('rx').value,+$('ry').value,+$('rz').value],move:[+$('mx').value,+$('my').value,+$('mz').value]};}
let transformBusy=false,transformPending=false;
async function applyTransform(){if(!modelData.length)return;if(transformBusy){transformPending=true;return;}transformBusy=true;try{do{transformPending=false;await post(`/api/model/${activeModel}/transform`,transformPayload());await refreshStatus();await refreshSource();$('transformStatus').textContent='Transform applied. Regenerate the mould to rebuild from this geometry.';$('transformStatus').dataset.kind='ok';}while(transformPending);}catch(e){$('transformStatus').textContent='Transform failed. '+e.message;$('transformStatus').dataset.kind='error';}finally{transformBusy=false;}}
transformIds.forEach(id=>$(id).addEventListener('input',()=>{if($('uniform').checked&&['sx','sy','sz'].includes(id)){['sx','sy','sz'].forEach(k=>$(k).value=$(id).value);}clearTimeout(transformTimer);transformTimer=setTimeout(applyTransform,180);}));$('uniform').onchange=()=>{};$('resetTransform').onclick=()=>{loadTransformDefaults();applyTransform();};
$('saveProject').onclick=async()=>{const r=await api('/api/project',{method:'POST'});const blob=await r.blob();downloadBlob(blob,'Mad_Hat_Project.zip');};
function downloadBlob(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href);}
$('openProject').onchange=async e=>{if(!e.target.files[0])return;const fd=new FormData();fd.append('file',e.target.files[0]);try{await api('/api/project/open',{method:'POST',body:fd});activeModel=0;await refreshStatus();await refreshSource();status('Project opened.','ok');}catch(err){status(err.message,'error');toast('Could not open project',err.message,'error');}e.target.value='';};
refreshStatus().then(refreshSource).catch(e=>status(e.message));
