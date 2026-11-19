from __future__ import annotations
import io, os, tempfile, zipfile, json, math, traceback
from pathlib import Path
import numpy as np
import trimesh
from flask import Flask, jsonify, request, send_file, render_template, abort
from engine import MoldGenerator, transform_to_export, transform_to_preview

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024

class Session:
    def __init__(self):
        self.models=[]       # {name, base, mesh, visible}
        self.generator=MoldGenerator()
        self.generated=False
        self.settings={}

S=Session()

def err(message, code=400):
    return jsonify({'ok':False,'error':str(message)}), code

def rebuild_source():
    visible=[m['mesh'].copy() for m in S.models if m['visible']]
    if visible:
        combined=trimesh.util.concatenate(visible)
        combined.remove_unreferenced_vertices(); combined.fix_normals()
        S.generator.source=combined
    else:
        S.generator.source=None
    S.generator.parts=[]; S.generator.plug=None; S.generated=False

def mesh_bytes(mesh, export=False):
    m=transform_to_export(mesh) if export else transform_to_preview(mesh)
    return m.export(file_type='stl')

def mesh_info(mesh):
    vol=abs(float(mesh.volume)) if mesh.is_watertight else 0.0
    return {'bounds':mesh.bounds.tolist(),'extents':mesh.extents.tolist(),
            'watertight':bool(mesh.is_watertight),'volume_mm3':vol,
            'volume_ml':vol/1000.0}

@app.get('/')
def index(): return render_template('index.html')

@app.get('/api/status')
def status():
    models=[]
    for i,m in enumerate(S.models):
        models.append({'index':i,'name':m['name'],'visible':m['visible'],**mesh_info(m['mesh'])})
    src=S.generator.source
    parts=[]
    if S.generated:
        parts=[{'index':i,'name':p.name,'direction':list(p.direction)} for i,p in enumerate(S.generator.parts)]
    return jsonify({'ok':True,'models':models,'parts':parts,'has_source':src is not None,
        'generated':S.generated,'source':mesh_info(src) if src is not None else None,
        'capacity':{'gap_ml':S.generator.cavity_volume_mm3/1000,
          'channel_ml':S.generator.pour_channel_volume_mm3/1000,
          'total_ml':S.generator.total_resin_volume_mm3/1000,
          'mix_ml':S.generator.total_resin_volume_mm3*1.1/1000}})

@app.post('/api/upload')
def upload():
    files=request.files.getlist('files')
    if not files: return err('Choose one or more STL files.')
    added=[]
    for f in files:
        if not f.filename: continue
        suffix=Path(f.filename).suffix.lower()
        if suffix != '.stl': return err(f'{f.filename}: only STL files are supported.')
        data=f.read()
        try:
            mesh=trimesh.load(io.BytesIO(data),file_type='stl',force='mesh')
            if not isinstance(mesh,trimesh.Trimesh) or len(mesh.faces)==0:
                raise ValueError('No triangle mesh found.')
            # Same source import convention as the desktop app: STL Z-up -> engine Y-up.
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2,[1,0,0]))
            mesh.remove_unreferenced_vertices(); mesh.fix_normals()
            mesh.apply_translation([-mesh.bounds.mean(axis=0)[0],-mesh.bounds[0,1],-mesh.bounds.mean(axis=0)[2]])
            if not mesh.is_watertight:
                raise ValueError('This source script requires a watertight STL. Repair the mesh and retry.')
            if S.models:
                existing_max_x=max(float(m['mesh'].bounds[1,0]) for m in S.models)
                mesh.apply_translation([existing_max_x+max(5.0,float(mesh.extents[0])*.10)-float(mesh.bounds[0,0]),0,0])
            S.models.append({'name':Path(f.filename).name,'base':mesh.copy(),'mesh':mesh.copy(),'visible':True})
            added.append(Path(f.filename).name)
        except Exception as e: return err(f'{f.filename}: {e}')
    rebuild_source()
    return jsonify({'ok':True,'added':added})

@app.post('/api/model/<int:index>/transform')
def transform(index):
    if index>=len(S.models): return err('Model not found.',404)
    d=request.get_json(force=True); m=S.models[index]['base'].copy()
    scale=d.get('scale',[1,1,1]); rot=d.get('rotate',[0,0,0]); move=d.get('move',[0,0,0])
    if len(scale)!=3 or len(rot)!=3 or len(move)!=3: return err('Transform values must contain X, Y and Z.')
    # Match the original desktop transform axis mapping exactly.
    m.apply_scale([float(scale[0]),float(scale[2]),float(scale[1])])
    centre=m.bounds.mean(axis=0)
    for angle,axis in ((rot[0],[1,0,0]),(rot[1],[0,0,-1]),(rot[2],[0,1,0])):
        if abs(float(angle))>1e-9:
            m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(float(angle)),axis,point=centre))
    m.apply_translation([-m.bounds.mean(axis=0)[0],-m.bounds[0,1],-m.bounds.mean(axis=0)[2]])
    m.apply_translation([float(move[0]),float(move[2]),-float(move[1])])
    m.fix_normals()
    if not m.is_watertight: return err('The transformed STL is no longer watertight.')
    S.models[index]['mesh']=m; rebuild_source()
    return jsonify({'ok':True})

@app.post('/api/model/<int:index>/visibility')
def visibility(index):
    if index>=len(S.models): return err('Model not found.',404)
    S.models[index]['visible']=bool(request.json.get('visible',True)); rebuild_source(); return jsonify({'ok':True})

@app.post('/api/model/<int:index>/duplicate')
def duplicate(index):
    if index>=len(S.models): return err('Model not found.',404)
    original=S.models[index]['mesh'].copy(); cp=original.copy()
    cp.apply_translation([float(original.extents[0])+max(5,float(original.extents[0])*.10),0,0])
    name=S.models[index]['name']+' (Copy)'
    S.models.append({'name':name,'base':cp.copy(),'mesh':cp.copy(),'visible':True}); rebuild_source()
    return jsonify({'ok':True})

@app.delete('/api/model/<int:index>')
def delete_model(index):
    if index>=len(S.models): return err('Model not found.',404)
    S.models.pop(index); rebuild_source(); return jsonify({'ok':True})

@app.post('/api/models/show-all')
def show_all():
    for m in S.models: m['visible']=True
    rebuild_source(); return jsonify({'ok':True})

@app.post('/api/generate')
def generate():
    if S.generator.source is None: return err('Load and show at least one STL first.')
    d=request.get_json(force=True)
    try:
        gen=MoldGenerator(); gen.source=S.generator.source.copy()
        style=d.get('style','multi')
        want_core=bool(d.get('core',False))
        if style=='flat':
            if want_core: gen.warnings.append('Core: not available for 2D flat pour moulds, so it was skipped.')
            gen.generate_flat_2d(wall=float(d.get('wall',8)),base=float(d.get('base',6)),
                top_extra=float(d.get('top_extra',8)),margin=float(d.get('margin',10)))
        else:
            gen.generate(panels=int(d.get('panels',4)),top_extra=float(d.get('top_extra',8)),
                wall=float(d.get('wall',8)),base=float(d.get('base',6)),
                clearance=float(d.get('clearance',3)),fill_diameter=float(d.get('fill_diameter',12)),
                fill_depth_percent=float(d.get('fill_depth_percent',50)),core=want_core,
                rim_width=float(d.get('rim_width',2.5)),
                rim_thickness=float(d.get('rim_thickness',1.5)))
        S.generator=gen; S.generated=True
        return jsonify({'ok':True,'parts':[p.name for p in gen.parts],'warnings':list(gen.warnings),
            'capacity':{'gap_ml':gen.cavity_volume_mm3/1000,'channel_ml':gen.pour_channel_volume_mm3/1000,
              'total_ml':gen.total_resin_volume_mm3/1000,'mix_ml':gen.total_resin_volume_mm3*1.1/1000}})
    except Exception as e:
        traceback.print_exc(); return err(e,500)

@app.get('/api/preview/source')
def preview_source():
    if S.generator.source is None:
        abort(404)

    # Before generation, show the imported source STL. Once generated, show
    # the exact plug mesh produced by the geometry engine. The plug already
    # includes its locating-pin holes and its final base-relative placement,
    # so the preview must not calculate or apply a second translation.
    preview_mesh = (
        S.generator.plug.copy()
        if S.generated and S.generator.plug is not None
        else S.generator.source.copy()
    )

    return send_file(
        io.BytesIO(mesh_bytes(preview_mesh)),
        mimetype='model/stl',
        download_name='Plug_Preview.stl' if S.generated else 'Source_Preview.stl'
    )

@app.get('/api/preview/part/<int:index>')
def preview_part(index):
    if not S.generated or index>=len(S.generator.parts): abort(404)
    return send_file(io.BytesIO(mesh_bytes(S.generator.parts[index].mesh)),mimetype='model/stl',download_name='Part_Preview.stl')

def base_and_source_mesh():
    """Return the mould base and positioned source STL as one export mesh.

    The flat 2D mould is already a single tray mesh with the source fused in,
    so it must not be combined a second time.
    """
    if not S.generated or not S.generator.parts:
        abort(404)
    if len(S.generator.parts) == 1:
        return S.generator.parts[0].mesh.copy()
    base = next((p.mesh for p in S.generator.parts if p.name == 'Mold_Base'), None)
    if base is None or S.generator.plug is None:
        abort(404)
    combined = trimesh.util.concatenate([base.copy(), S.generator.plug.copy()])
    combined.remove_unreferenced_vertices()
    combined.fix_normals()
    return combined

@app.get('/api/export/part/<int:index>')
def export_part(index):
    if not S.generated or index>=len(S.generator.parts): abort(404)
    part = S.generator.parts[index]
    mesh = base_and_source_mesh() if part.name == 'Mold_Base' else part.mesh.copy()
    data=mesh_bytes(mesh,True)
    filename = 'Mold_Base_and_STL.stl' if part.name == 'Mold_Base' else part.name+'.stl'
    return send_file(io.BytesIO(data),mimetype='application/sla',as_attachment=True,download_name=filename)

@app.get('/api/export/plug')
def export_plug():
    if not S.generated or S.generator.plug is None: abort(404)
    data=mesh_bytes(base_and_source_mesh(),True)
    return send_file(io.BytesIO(data),mimetype='application/sla',as_attachment=True,download_name='Mold_Base_and_STL.stl')

@app.get('/api/export/zip')
def export_zip():
    if not S.generated: abort(404)
    amount=max(0,min(100,float(request.args.get('explode',0))))
    max_size=max(S.generator.source.extents); dist=(amount/100)*(8+max(12,max_size*.12))
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,'w',zipfile.ZIP_DEFLATED) as zf:
        for i,p in enumerate(S.generator.parts):
            if p.name == 'Mold_Base':
                mesh = base_and_source_mesh()
                filename = 'Mold_Base_and_STL.stl'
            else:
                mesh = p.mesh.copy()
                filename = p.name+'.stl'
            if i and amount:
                mesh.apply_translation(np.array(p.direction)*dist)
            zf.writestr(filename,transform_to_export(mesh).export(file_type='stl'))
    buff.seek(0)
    return send_file(buff,mimetype='application/zip',as_attachment=True,download_name='Mad_Hat_Mold_Parts.zip')

@app.post('/api/project')
def project_save():
    # Project archive stores original imported STLs plus current transforms/visibility.
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,'w',zipfile.ZIP_DEFLATED) as z:
        meta=[]
        for i,m in enumerate(S.models):
            z.writestr(f'models/model_{i+1:03d}.stl',m['mesh'].export(file_type='stl'))
            meta.append({'name':m['name'],'visible':m['visible']})
        z.writestr('project.json',json.dumps({'models':meta},indent=2))
    buff.seek(0)
    return send_file(buff,mimetype='application/zip',as_attachment=True,download_name='Mad_Hat_Project.zip')

@app.post('/api/project/open')
def project_open():
    f=request.files.get('file')
    if not f: return err('Choose a project ZIP.')
    try:
        z=zipfile.ZipFile(io.BytesIO(f.read()))
        meta=json.loads(z.read('project.json'))
        models=[]
        for i,item in enumerate(meta['models']):
            data=z.read(f'models/model_{i+1:03d}.stl')
            mesh=trimesh.load(io.BytesIO(data),file_type='stl',force='mesh')
            # Project models are stored in the engine's native Y-up coordinates.
            mesh.remove_unreferenced_vertices(); mesh.fix_normals()
            if not mesh.is_watertight: raise ValueError(f"{item['name']} is not watertight")
            models.append({'name':item['name'],'base':mesh.copy(),'mesh':mesh.copy(),'visible':bool(item.get('visible',True))})
        S.models=models; rebuild_source()
        return jsonify({'ok':True})
    except Exception as e: return err(f'Could not open project: {e}')

if __name__=='__main__':
    print('\nMAD HAT Negative Mold Generator — Web Interface')
    print('Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.\n')
    app.run(host='127.0.0.1',port=5000,debug=False,threaded=True)
