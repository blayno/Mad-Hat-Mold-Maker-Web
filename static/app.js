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
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.localClippingEnabled=true;const clipPlanes=[];renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('canvas').appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,0,40);
scene.add(new THREE.HemisphereLight(0xffffff,0x47515c,2));
const key=new THREE.DirectionalLight(0xffffff,2.2);key.position.set(150,-180,260);scene.add(key);
const fill=new THREE.DirectionalLight(0x9bcfff,.8);fill.position.set(-180,100,120);scene.add(fill);
let bedGrid=null,bedOutline=null;
function updateBedGrid(){
  // Build a rectangular 10 mm grid matching the configured printer bed.
  const bx=Math.max(1,Number($('bed_x')?.value)||220),by=Math.max(1,Number($('bed_y')?.value)||220);
  if(bedGrid){scene.remove(bedGrid);bedGrid.geometry.dispose();bedGrid.material.dispose();bedGrid=null;}
  if(bedOutline){scene.remove(bedOutline);bedOutline.geometry.dispose();bedOutline.material.dispose();bedOutline=null;}
  const positions=[];const step=10;
  const add=(x1,y1,x2,y2)=>positions.push(x1,y1,0,x2,y2,0);
  for(let x=-bx/2;x<=bx/2+1e-6;x+=step)add(x,-by/2,x,by/2);
  for(let y=-by/2;y<=by/2+1e-6;y+=step)add(-bx/2,y,bx/2,y);
  // Ensure the outermost lines exactly match the bed dimensions.
  add(-bx/2,-by/2,-bx/2,by/2);add(bx/2,-by/2,bx/2,by/2);
  add(-bx/2,-by/2,bx/2,-by/2);add(-bx/2,by/2,bx/2,by/2);
  const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  bedGrid=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({color:0x354250}));scene.add(bedGrid);
  const outlineGeo=new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-bx/2,-by/2,.15),new THREE.Vector3(bx/2,-by/2,.15),
    new THREE.Vector3(bx/2,by/2,.15),new THREE.Vector3(-bx/2,by/2,.15)
  ]);
  bedOutline=new THREE.LineLoop(outlineGeo,new THREE.LineBasicMaterial({color:0xf0a500}));scene.add(bedOutline);
}
scene.add(new THREE.AxesHelper(70));

const loader=new STLLoader();
let sourceMesh=null,partMeshes=[],partData=[],modelData=[],activeModel=0,selectedPart=0,sourceVisible=true,explode=0,sourceMaxSize=100,generated=false,flatMode=false,generating=false,hasSource=false,lastStatus=null;
let sourceToken=0,partsToken=0,cutPlaneMeshes=[];
const NOTE_3D='Includes the silicone gap and pour channel without double-counting their overlap.',NOTE_2D='Silicone fills the contoured well up to the top of the wall, minus the volume the original STL displaces.';
const PART_COLORS=['#e34b91','#9b58e8','#d5ae45','#73c95d','#39b8a9','#8793a8','#e07848','#6b9fe0'];
const IDENTITY_TF={scale:[1,1,1],rotate:[0,0,0],move:[0,0,0]};

function mat(color,rough=.7){return new THREE.MeshStandardMaterial({color,roughness:rough,metalness:.05,side:THREE.DoubleSide,clippingPlanes:clipPlanes});}
function clearParts(){for(const m of partMeshes){scene.remove(m);m.geometry.dispose();}partMeshes=[];partData=[];}

// Preview the exact radial split planes used by the geometry engine.
function clearCutPlanes(){for(const m of cutPlaneMeshes){scene.remove(m);m.geometry.dispose();m.material.dispose();}cutPlaneMeshes=[];}
function cutPlaneOffsets(){
  try{const a=JSON.parse($('cut_offsets')?.value||'[]');return Array.isArray(a)?a.map(v=>Array.isArray(v)?[Number(v[0])||0,Number(v[1])||0]:[0,0]):[];}catch(e){return [];}
}
function panelCount(){return Math.max(2,Math.min(8,Number($('panels')?.value)||4));}
// A full vertical cut plane creates two opposing panel seams. Even panel
// counts therefore need half as many planes; the legacy 5-panel option keeps
// five independent radial seams because it cannot be made from diametral pairs.
function cutPlaneCount(){const n=panelCount();return n%2===0?n/2:n;}
function cutAnglesDegrees(){
  const n=panelCount(),count=cutPlaneCount();
  return Array.from({length:count},(_,i)=>i*(n%2===0?180:360)/count);
}
function syncCutPlaneControls(){
  const count=cutPlaneCount();
  let offsets=cutPlaneOffsets();if(offsets.length!==count)offsets=Array.from({length:count},()=>[0,0]);
  $('cut_offsets').value=JSON.stringify(offsets);
  const box=$('cutPlaneControls');box.innerHTML='';
  offsets.forEach((v,i)=>{
    const group=document.createElement('div');group.className='field';
    const title=document.createElement('span');title.className='lbl';title.textContent=`Plane ${i+1} position`;
    const row=document.createElement('div');row.className='grid2';
    [['X',0],['Y',1]].forEach(([axis,idx])=>{
      const wrap=document.createElement('label');wrap.className='field';
      const lbl=document.createElement('span');lbl.className='lbl';lbl.textContent=axis+' movement';
      const inp=document.createElement('span');inp.className='inp';
      const input=document.createElement('input');input.type='number';input.step='0.5';input.min='-200';input.max='200';input.value=(v[idx]||0).toFixed(1);input.setAttribute('aria-label',`Plane ${i+1} ${axis} movement in mm`);
      const unit=document.createElement('em');unit.textContent='mm';
      input.addEventListener('input',()=>{const next=cutPlaneOffsets();next[i][idx]=Math.max(-200,Math.min(200,Number(input.value)||0));$('cut_offsets').value=JSON.stringify(next);refreshCutPlanePositions();saveSettings();});
      inp.append(input,unit);wrap.append(lbl,inp);row.append(wrap);
    });
    group.append(title,row);box.append(group);
  });
}
function refreshCutPlanePositions(){
  if(!sourceMesh||!cutPlaneMeshes.length)return;
  const offsets=cutPlaneOffsets();const box=new THREE.Box3().setFromObject(sourceMesh);const center=box.getCenter(new THREE.Vector3());
  cutPlaneMeshes.forEach((mesh,i)=>{const v=offsets[i]||[0,0];mesh.position.x=center.x+v[0];mesh.position.y=center.y+v[1];});
}
function updateCutPlanes(){
  clearCutPlanes();
  if(!sourceMesh||generated||flatMode||$('style')?.value!=='multi'||!$('showCutPlanes')?.checked)return;
  const angles=cutAnglesDegrees(),count=angles.length;
  const box=new THREE.Box3().setFromObject(sourceMesh),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
  const radius=Math.max(size.x,size.y,size.z,10)*1.25;
  const base=Math.max(0,Number($('base')?.value)||6),extra=Math.max(0,Number($('top_extra')?.value)||8);
  const height=size.z+base+extra;
  const planeCenter=new THREE.Vector3(center.x,center.y,(box.min.z+box.max.z+extra+base)/2);
  const offsets=cutPlaneOffsets();
  const material=new THREE.MeshBasicMaterial({color:0xff2525,transparent:true,opacity:.24,side:THREE.DoubleSide,depthWrite:false});
  for(let i=0;i<count;i++){
    const a=angles[i]*Math.PI/180;
    const radial=new THREE.Vector3(Math.sin(a),-Math.cos(a),0).normalize();
    const vertical=new THREE.Vector3(0,0,1);
    const normal=new THREE.Vector3().crossVectors(radial,vertical).normalize();
    const basis=new THREE.Matrix4().makeBasis(radial,vertical,normal);
    const mesh=new THREE.Mesh(new THREE.PlaneGeometry(radius*2,height),material.clone());
    mesh.quaternion.setFromRotationMatrix(basis);
    const v=offsets[i]||[0,0];mesh.position.copy(planeCenter).add(new THREE.Vector3(v[0],v[1],0));
    mesh.renderOrder=5;mesh.userData.cutPlaneIndex=i;scene.add(mesh);cutPlaneMeshes.push(mesh);
  }
}

// Cut planes are intentionally NOT draggable in the 3D preview. The preview
// remains dedicated to orbit/pan/zoom; plane positions are changed only through
// the X/Y numeric settings above and reflected here by refreshCutPlanePositions().

// Creased normals: smooth shading on curved walls, crisp edges on flanges and caps.
function addSTL(url,material){return new Promise((resolve,reject)=>loader.load(url,g=>{
  let geo=g;
  try{if(g.attributes.position.count<1200000)geo=toCreasedNormals(g,Math.PI/5);else g.computeVertexNormals();}
  catch(e){g.computeVertexNormals();geo=g;}
  if(geo!==g)g.dispose();
  resolve(new THREE.Mesh(geo,material));
},undefined,reject));}

function sceneBox(){const box=new THREE.Box3();if(sourceMesh)box.expandByObject(sourceMesh);partMeshes.forEach(m=>{if(m.visible)box.expandByObject(m)});return box;}
function updateInfo(){updateCut();const box=sceneBox();if(box.isEmpty()){$('sceneInfo').textContent='';return;}const size=box.getSize(new THREE.Vector3()),f=v=>v.toFixed(1);$('sceneInfo').innerHTML=`<span style="color:#ff5c5c;font-weight:600">X</span> ${f(size.x)} × <span style="color:#5ce05c;font-weight:600">Y</span> ${f(size.y)} × <span style="color:#5c9bff;font-weight:600">Z</span> ${f(size.z)} mm`;}
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
const SETTING_FIELDS=['bed_x','bed_y','bed_z','filament_density','print_speed','layer_height','resin_density','silicone_density','style','panels','wall_style','margin','top_extra','wall','base','clearance','fill_diameter','fill_depth_percent','rim_width','rim_thickness','key_width','key_depth','cut_offsets','project_name'];
const SETTING_CHECKS=['core_enabled','label_tab','showCutPlanes'];
const SETTINGS_KEY='madhat.settings.v1';
function getSettings(){const s={};SETTING_FIELDS.forEach(id=>s[id]=$(id).value);SETTING_CHECKS.forEach(id=>s[id]=$(id).checked);return s;}
function applySettings(s){if(!s||typeof s!=='object')return;SETTING_FIELDS.forEach(id=>{if(s[id]!==undefined&&s[id]!==null)$(id).value=s[id];});SETTING_CHECKS.forEach(id=>{if(typeof s[id]==='boolean')$(id).checked=s[id];});syncCore();syncName();syncSilicone();if($('cutPlaneControls'))syncCutPlaneControls();}
function saveSettings(){try{localStorage.setItem(SETTINGS_KEY,JSON.stringify(getSettings()));}catch(e){}}
function restoreSettings(){try{applySettings(JSON.parse(localStorage.getItem(SETTINGS_KEY)||'null'));}catch(e){}}

// ------------------------------------------------------------------ presets
// Named setting sets live in this browser (localStorage). They hold the mould
// settings only, never the project / file name.
const PRESETS_KEY='madhat.presets.v1';
const presetSettings=()=>{const s=getSettings();delete s.project_name;return s;};
const DEFAULT_SETTINGS=presetSettings();
function loadPresets(){try{const p=JSON.parse(localStorage.getItem(PRESETS_KEY)||'[]');return Array.isArray(p)?p.filter(x=>x&&typeof x.name==='string'&&x.settings&&typeof x.settings==='object'):[];}catch(e){return [];}}
function storePresets(list){try{localStorage.setItem(PRESETS_KEY,JSON.stringify(list));return true;}catch(e){toast('Could not save the preset','This browser is blocking local storage.','error');return false;}}
function renderPresets(selected=''){
  const sel=$('presetSelect');sel.innerHTML='';
  const add=(v,t)=>{const o=document.createElement('option');o.value=v;o.textContent=t;sel.append(o);};
  add('','Current settings');add('__default__','Factory defaults');
  loadPresets().sort((a,b)=>a.name.localeCompare(b.name)).forEach(p=>add('u:'+p.name,p.name));
  sel.value=[...sel.options].some(o=>o.value===selected)?selected:'';
  $('presetDelete').disabled=!sel.value.startsWith('u:');
}
$('presetSelect').onchange=()=>{
  const v=$('presetSelect').value;let s=null;
  if(v==='__default__')s=DEFAULT_SETTINGS;
  else if(v.startsWith('u:'))s=(loadPresets().find(p=>'u:'+p.name===v)||{}).settings;
  if(s){applySettings({...s});saveSettings();}
  $('presetDelete').disabled=!v.startsWith('u:');
};
$('presetSave').onclick=()=>{
  const current=$('presetSelect').value;
  const name=(prompt('Name this preset:',current.startsWith('u:')?current.slice(2):'')||'').trim().slice(0,40);
  if(!name)return;
  const list=loadPresets(),i=list.findIndex(p=>p.name===name);
  if(i>=0&&!confirm(`A preset called "${name}" already exists. Replace it?`))return;
  const entry={name,settings:presetSettings()};
  if(i>=0)list[i]=entry;else list.push(entry);
  if(storePresets(list)){renderPresets('u:'+name);toast('Preset saved',name,'ok');}
};
$('presetDelete').onclick=()=>{
  const v=$('presetSelect').value;if(!v.startsWith('u:'))return;
  const name=v.slice(2);if(!confirm(`Delete the preset "${name}"?`))return;
  if(storePresets(loadPresets().filter(p=>p.name!==name)))renderPresets('');
};
// Editing any mould setting means the dropdown no longer describes what is on screen.
[...SETTING_FIELDS,...SETTING_CHECKS].filter(id=>id!=='project_name').forEach(id=>$(id).addEventListener('change',()=>{$('presetSelect').value='';$('presetDelete').disabled=true;}));

// Silicone thickness shortcuts set the cavity clearance (the gap the silicone fills).
function syncSilicone(){const v=+$('clearance').value;document.querySelectorAll('#siliconeSeg button').forEach(b=>b.classList.toggle('selected',Math.abs(+b.dataset.clear-v)<1e-6));}
document.querySelectorAll('#siliconeSeg button').forEach(b=>b.onclick=()=>{$('clearance').value=b.dataset.clear;['input','change'].forEach(t=>$('clearance').dispatchEvent(new Event(t,{bubbles:true})));syncSilicone();});
$('clearance').addEventListener('input',syncSilicone);

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
  ['saveProject','deleteAll','showAll','deleteSelected','duplicate','centerMould'].forEach(id=>$(id).disabled=!d.models.length);
  lastStatus=d;renderStats(d);updateCutPlanes();
  if(d.generated){
    await renderParts(d.parts);
  }else{
    $('partsList').textContent='Generate a mould to see its parts.';
    if($('printStatsLabel')){$('printStatsLabel').hidden=true;}
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
    {const w=d.flat?null:d.capacity.min_wall_mm,el=$('thinWall');
      if(w==null){el.textContent=dash;el.style.color='';$('thinWallNote').textContent=d.flat?'Not applicable to 2D flat pour moulds.':'Only estimated for the "follows the model" outer wall.';}
      else{el.textContent=`~${w.toFixed(1)} mm`;el.style.color=w<2?'#ff6b6b':w<3?'#e0a030':'';
        $('thinWallNote').textContent=`Thinnest spot is at ${d.capacity.min_wall_where}. Estimated from the outer wall, top and pour hole; flanges, rim groove and core bore are not included.`+(w<2?' Under ~2 mm may print poorly or break.':'');}}
  }else{
    $('flatDims').hidden=true;$('siliconeNote').textContent=NOTE_3D;
    $('gapL').textContent='Gap around the model';$('channelL').textContent='Pour channel';$('thinWall').textContent=dash;$('thinWall').style.color='';
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
  updateCutPlanes();
  updateInfo();
}

function printEstimate(part){
  const bed=[+$('bed_x').value,+$('bed_y').value,+$('bed_z').value];
  const size=part.size_mm||[0,0,0];
  const fits=size[2]<=bed[2]+0.01 && ((size[0]<=bed[0]+0.01&&size[1]<=bed[1]+0.01)||(size[0]<=bed[1]+0.01&&size[1]<=bed[0]+0.01));
  const density=+$('filament_density').value||1.24;
  const speed=+$('print_speed').value||50;
  const layer=+$('layer_height').value||0.2;
  const volume=Math.max(0,+part.volume_mm3||0);
  const grams=volume/1000*density;
  // Approximate extrusion time from deposited volume / volumetric flow, plus 30% overhead.
  const flow=0.4*layer*speed;
  const minutes=flow>0?(volume/flow/60)*1.30:0;
  const duration=minutes>=60?`${(minutes/60).toFixed(1)} hr`:`${Math.max(1,Math.round(minutes))} min`;
  return {fits,grams,duration,size,bed};
}
function updatePrintability(){
  if(!lastStatus?.generated)return;
  document.querySelectorAll('#partsList .listitem[data-part-index]').forEach(row=>{
    const part=lastStatus.parts.find(p=>p.index===+row.dataset.partIndex);if(!part)return;
    const e=printEstimate(part),meta=row.querySelector('.print-meta');if(!meta)return;
    meta.innerHTML=`<span style="color:${e.fits?'#73c95d':'#ff6b6b'};font-weight:600">${e.fits?'FITS BED':'TOO LARGE'}</span> · ${e.grams.toFixed(1)} g · ~${e.duration}`;
    meta.title=`Part ${e.size.map(v=>v.toFixed(1)).join(' × ')} mm; bed ${e.bed.join(' × ')} mm`;
  });
  updatePrintStatsLabel();
}
function updatePrintStatsLabel(){
  const el=$('printStatsLabel');if(!el)return;
  if(!lastStatus?.generated||!lastStatus.parts?.length){el.hidden=true;el.textContent='';return;}
  const parts=lastStatus.parts,est=parts.map(printEstimate);
  const fit=est.filter(e=>e.fits).length,grams=est.reduce((a,e)=>a+e.grams,0);
  // Single-plate estimate: every part printed together in the same layers, so layer changes
  // are paid once for the tallest part instead of once per part.
  const layer=+$('layer_height').value||0.2,speed=+$('print_speed').value||50,flow=0.4*layer*speed;
  const totalVol=parts.reduce((a,p)=>a+Math.max(0,+p.volume_mm3||0),0);
  const tallest=Math.max(...parts.map(p=>(p.size_mm||[0,0,0])[2]||0));
  const SEC_PER_LAYER=4,EXTRUDE_OVERHEAD=1.15;   // rough travel/accel and layer-change allowances
  const secs=flow>0?(totalVol/flow)*EXTRUDE_OVERHEAD+Math.ceil(tallest/layer)*SEC_PER_LAYER:0;
  const mins=secs/60;
  const dur=mins>=60?`${(mins/60).toFixed(1)} hr`:`${Math.max(1,Math.round(mins))} min`;
  // Rough single-plate fit: tallest part within bed Z and combined footprints (+10% spacing) within bed area.
  const bx=+$('bed_x').value,by=+$('bed_y').value,bz=+$('bed_z').value;
  const area=parts.reduce((a,p)=>{const z=p.size_mm||[0,0,0];return a+z[0]*z[1];},0)*1.1;
  const onePlate=est.every(e=>e.fits)&&tallest<=bz+0.01&&area<=bx*by;
  el.hidden=false;
  el.innerHTML=`Print stats: <span style="color:${onePlate?'#73c95d':'#ff6b6b'};font-weight:600">${onePlate?'1 PLATE':'WON\'T FIT 1 PLATE'}</span> · ${grams.toFixed(1)} g · ~${dur}`;
  el.title=`All ${parts.length} parts printed together on one plate (${fit}/${parts.length} fit the bed individually). Time is an estimate.`;
}
['bed_x','bed_y','bed_z','filament_density','print_speed','layer_height'].forEach(id=>$(id).addEventListener('input',()=>{saveSettings();updatePrintability();if(id==='bed_x'||id==='bed_y')updateBedGrid();}));

async function renderParts(parts){
  const token=++partsToken;clearParts();partData=parts;
  const list=$('partsList');list.innerHTML='';$('selectedPart').innerHTML='';
  for(const p of parts){
    const m=await addSTL(`/api/preview/part/${p.index}?${Date.now()}`,mat(PART_COLORS[p.index%8]));
    if(token!==partsToken){m.geometry.dispose();return;}
    m.userData.basePosition=m.position.clone();scene.add(m);partMeshes.push(m);
    const row=document.createElement('div');row.className='listitem';row.dataset.partIndex=p.index;
    const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;cb.setAttribute('aria-label',`Show ${p.name}`);cb.onchange=()=>m.visible=cb.checked;
    const sw=document.createElement('i');sw.className='swatch';sw.style.background=PART_COLORS[p.index%8];
    const label=document.createElement('span');label.textContent=p.name;
    row.append(cb,sw,label);
    const meta=document.createElement('small');meta.className='muted print-meta';meta.style.cssText='display:block;margin-left:auto;text-align:right';row.append(meta);
    if(p.size_mm){row.title='Printed size: '+p.size_mm.map(v=>v.toFixed(1)).join(' × ')+' mm';}
    list.append(row);
    const opt=document.createElement('option');opt.value=p.index;opt.textContent=p.name;$('selectedPart').append(opt);
  }
  applyExplode();updateInfo();updatePrintability();
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
  const d={style:$('style').value,panels:+$('panels').value,cut_angles:Array.from({length:panelCount()},(_,i)=>i*360/panelCount()),cut_positions:cutPlaneOffsets().map(([x,y])=>[x,-y]),wall_style:$('wall_style').value,top_extra:+$('top_extra').value,wall:+$('wall').value,base:+$('base').value,clearance:+$('clearance').value,fill_diameter:+$('fill_diameter').value,fill_depth_percent:+$('fill_depth_percent').value,rim_width:+$('rim_width').value,rim_thickness:+$('rim_thickness').value,key_width:+$('key_width').value,key_depth:+$('key_depth').value,margin:+$('margin').value,label_tab:$('label_tab').checked,core:$('core_enabled').checked&&$('style').value!=='flat'};
  saveSettings();
  try{
    await post('/api/generate',d);
    await finishJob(await watchJob());
  }catch(e){status('Generation failed. '+e.message,'error');toast('Generation failed',e.message,'error');}
};
$('cancelGen').onclick=async()=>{$('cancelGen').disabled=true;try{await post('/api/generate/cancel',{});}catch(e){}};

const syncCore=()=>{
  const flat=$('style').value==='flat';
  document.body.dataset.style=flat?'flat':'multi';
  // Explicitly show only the controls belonging to the selected mould mode.
  document.querySelectorAll('[data-only]').forEach(el=>{el.hidden=el.dataset.only!==(flat?'flat':'multi');});
  $('core_enabled').disabled=flat;$('label_tab').disabled=!flat;
  if(flat)$('core_enabled').checked=false;
  // Remove the red cut-plane overlays immediately when switching to 2D mode.
  updateCutPlanes();
};
$('style').addEventListener('change',syncCore);
syncCore();
[...SETTING_FIELDS,...SETTING_CHECKS].forEach(id=>$(id).addEventListener('change',saveSettings));
$('panels').addEventListener('change',()=>{syncCutPlaneControls();updateCutPlanes();saveSettings();});
['style','base','top_extra','showCutPlanes'].forEach(id=>{
  $(id).addEventListener('input',updateCutPlanes);
  $(id).addEventListener('change',updateCutPlanes);
});

// -------------------------------------------------------------- view/export
$('showSource').onchange=()=>{sourceVisible=$('showSource').checked;if(sourceMesh)sourceMesh.visible=sourceVisible;};
function syncExplodeButton(){$('explodeBtn').textContent=explode>0?'ASSEMBLE MOLD':'EXPLODE MOLD';}
$('explode').oninput=()=>{explode=+$('explode').value;$('explodeLabel').textContent=explode+'%';syncExplodeButton();applyExplode();};
$('explodeBtn').onclick=()=>{$('explode').value=explode>0?0:100;$('explode').dispatchEvent(new Event('input'));};
syncExplodeButton();
$('selectedPart').onchange=()=>selectedPart=+$('selectedPart').value;
// File names come from the Project / file name field. Where the browser supports it
// (Chrome, Edge) a Save dialog opens first so the name and folder can be changed;
// otherwise the file downloads with the typed name.
const cleanName=(s,fallback)=>{s=String(s||'').trim().replace(/[\\/:*?"<>|\u0000-\u001f]/g,'').replace(/\s+/g,'_').replace(/^\.+/,'').trim();return (s||fallback).slice(0,80);};
const projectBase=()=>cleanName($('project_name').value,'Mad_Hat');
function syncName(){$('project_name2').value=$('project_name').value;}
$('project_name').addEventListener('input',syncName);
$('project_name2').addEventListener('input',()=>{$('project_name').value=$('project_name2').value;saveSettings();});
syncName();
async function saveAs(filename,mime,produce){
  const ext=filename.slice(filename.lastIndexOf('.'));
  let handle=null;
  if(window.showSaveFilePicker){
    try{handle=await window.showSaveFilePicker({suggestedName:filename,types:[{description:ext==='.zip'?'ZIP archive':'STL mesh',accept:{[mime]:[ext]}}]});}
    catch(e){if(e&&e.name==='AbortError')return false;handle=null;}
  }
  const blob=await produce();
  if(handle){const w=await handle.createWritable();await w.write(blob);await w.close();}
  else downloadBlob(blob,filename);
  return true;
}
const fetchBlob=async url=>(await api(url)).blob();
const attempt=(what,fn)=>async()=>{try{await fn();}catch(e){toast('Could not '+what,e.message,'error');}};
$('exportSelected').onclick=attempt('export the part',async()=>{
  if(!partMeshes.length)return;
  const opt=$('selectedPart').selectedOptions[0];const raw=opt?opt.textContent:'part';
  const part=raw==='Mold_Base'?'Mold_Base_and_STL':cleanName(raw,'part');
  await saveAs(`${projectBase()}_${part}.stl`,'model/stl',()=>fetchBlob(`/api/export/part/${$('selectedPart').value}`));
});
$('exportAll').onclick=attempt('export the parts',()=>saveAs(`${projectBase()}_parts.zip`,'application/zip',()=>fetchBlob('/api/export/zip')));
$('exportExploded').onclick=attempt('export the exploded mould',()=>saveAs(`${projectBase()}_exploded.zip`,'application/zip',()=>fetchBlob('/api/export/zip?explode='+explode)));
$('exportPlug').onclick=attempt('export the base and STL',()=>saveAs(`${projectBase()}_base_and_STL.stl`,'model/stl',()=>fetchBlob('/api/export/plug')));

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
$('centerMould').onclick=async()=>{
  if(!modelData.length)return;
  try{
    await post('/api/center',{});
    await refreshStatus();await refreshSource();syncTransformFields();fitCamera();
    $('transformStatus').textContent='Mould and models centred on the build plate.';$('transformStatus').dataset.kind='ok';
  }catch(e){$('transformStatus').textContent='Centre failed. '+e.message;$('transformStatus').dataset.kind='error';}
};
$('resetTransform').onclick=()=>{transformIds.forEach(id=>$(id).value=(id[0]==='s'?1:0));applyTransform();};

// ------------------------------------------------------------------ project
function downloadBlob(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href);}
$('saveProject').onclick=attempt('save the project',()=>saveAs(`${projectBase()}.zip`,'application/zip',async()=>(await post('/api/project',{settings:getSettings()})).blob()));
$('openProject').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  $('project_name').value=cleanName(file.name.replace(/\.zip$/i,''),'Mad_Hat');syncName();
  const fd=new FormData();fd.append('file',file);
  try{
    const res=await api('/api/project/open',{method:'POST',body:fd});
    applySettings(res.settings);saveSettings();
    activeModel=0;await refreshStatus();syncTransformFields();await refreshSource();fitCamera();status('Project opened.','ok');
  }catch(err){status(err.message,'error');toast('Could not open project',err.message,'error');}
  e.target.value='';
};


// ----------------------------------------------------------------- cutaway
// A clipping plane shared by every part material (see mat()). Z is up here.
function updateCut(){
  clipPlanes.length=0;
  const on=$('cutOn').checked;
  ['cutAxis','cut','cutFlip'].forEach(id=>$(id).disabled=!on);
  if(!on){$('cutLabel').textContent='—';return;}
  const box=sceneBox();if(box.isEmpty())return;
  const ax=$('cutAxis').value,i={x:0,y:1,z:2}[ax];
  const lo=box.min.getComponent(i),hi=box.max.getComponent(i);
  const pos=lo+(hi-lo)*(+$('cut').value/100);
  const sign=(ax==='z'?-1:1)*($('cutFlip').checked?-1:1);   // default: remove the side nearest the default camera
  const n=new THREE.Vector3();n.setComponent(i,sign);
  clipPlanes.push(new THREE.Plane(n,-sign*pos));
  $('cutLabel').textContent=pos.toFixed(1)+' mm';
}
['cutOn','cutAxis','cutFlip'].forEach(id=>$(id).addEventListener('change',updateCut));
$('cut').addEventListener('input',updateCut);

// -------------------------------------------------------------------- start
restoreSettings();syncSilicone();syncCutPlaneControls();renderPresets('');updateBedGrid();updateCut();
(async()=>{
  try{
    await refreshStatus();syncTransformFields();await refreshSource();fitCamera();
    // If the page was reloaded while a mould was building, pick the job back up.
    const j=await api('/api/job');
    if(j.state==='running')await finishJob(await watchJob());
  }catch(e){status(e.message);}
})();
