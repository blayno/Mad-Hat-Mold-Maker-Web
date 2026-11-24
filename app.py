from __future__ import annotations
import io, zipfile, json, math, traceback, threading, time, uuid
from functools import wraps
from pathlib import Path
import numpy as np
import trimesh
from flask import Flask, jsonify, request, send_file, render_template, abort
from engine import (MoldGenerator, transform_to_export, transform_to_preview,
                    transform_set_to_export)

app = Flask(__name__)
import vendor                      # three.js + font served locally (CDN fallback)
vendor.register(app)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024

MESH_EXTENSIONS = {'.stl', '.obj', '.3mf'}
MAX_PROJECT_UNCOMPRESSED = 2 * 1024 ** 3   # refuse zip bombs
IDENTITY_TF = {'scale': [1.0, 1.0, 1.0], 'rotate': [0.0, 0.0, 0.0], 'move': [0.0, 0.0, 0.0]}

# One re-entrant lock guards every read or write of the session. Generation runs
# in a background thread and only takes the lock to snapshot its inputs and to
# commit its result.
LOCK = threading.RLock()


def locked(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        with LOCK:
            return fn(*a, **k)
    return wrapper


class Session:
    def __init__(self):
        # model: {name, base, mesh, tf, visible, info}
        #   base = the mesh as imported / laid out (never changes after import)
        #   tf   = scale / rotate / move applied to base to give `mesh`
        self.models = []
        self.generator = MoldGenerator()
        self.generated = False
        self.rev = 0           # bumped whenever the source changes


S = Session()


def err(message, code=400):
    return jsonify({'ok': False, 'error': str(message)}), code


# ---------------------------------------------------------------- validation
def require_watertight(mesh, label='Mesh'):
    if not mesh.is_watertight:
        raise ValueError(f'{label} is not watertight (it has holes or open edges). Repair the mesh and try again.')
    if not mesh.is_winding_consistent:
        raise ValueError(f'{label} has inconsistent face winding. Repair the mesh and try again.')
    # abs(): an inside-out mesh has negative volume but is otherwise fine.
    if abs(float(mesh.volume)) < 1e-9:
        raise ValueError(f'{label} has zero volume. Repair the mesh and try again.')


def clean_tf(tf):
    tf = tf or {}
    out = {}
    for key in ('scale', 'rotate', 'move'):
        vals = tf.get(key, IDENTITY_TF[key])
        if not isinstance(vals, (list, tuple)) or len(vals) != 3:
            raise ValueError('Transform values must contain X, Y and Z.')
        vals = [float(v) for v in vals]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError('Transform values must be finite numbers.')
        out[key] = vals
    if any(v <= 0 for v in out['scale']):
        raise ValueError('Scale must be greater than zero.')
    return out


def is_identity(tf):
    return (all(abs(v - 1) < 1e-12 for v in tf['scale']) and
            all(abs(v) < 1e-12 for v in tf['rotate']) and
            all(abs(v) < 1e-12 for v in tf['move']))


def apply_tf(base, tf):
    """Scale, rotate and move `base` exactly as the original app did.

    The result is re-centred on X/Z = 0 with its bottom on Y = 0, and tf['move']
    is then added, so a transformed model is positioned from the world origin.
    """
    scale, rot, move = tf['scale'], tf['rotate'], tf['move']
    m = base.copy()
    # Axis mapping matches the original desktop transform (engine is Y-up).
    m.apply_scale([scale[0], scale[2], scale[1]])
    centre = m.bounds.mean(axis=0)
    for angle, axis in ((rot[0], [1, 0, 0]), (rot[1], [0, 0, -1]), (rot[2], [0, 1, 0])):
        if abs(angle) > 1e-9:
            m.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(angle), axis, point=centre))
    m.apply_translation([-m.bounds.mean(axis=0)[0], -m.bounds[0, 1], -m.bounds.mean(axis=0)[2]])
    m.apply_translation([move[0], move[2], -move[1]])
    m.fix_normals()
    return m


def make_model(name, base, tf=None, visible=True, applied=False):
    """`applied` is False until a transform has been run on the model; until then
    the mesh is just the imported/laid-out base, as in the original app."""
    tf = clean_tf(tf) if tf else clean_tf(IDENTITY_TF)
    mesh = apply_tf(base, tf) if applied else base.copy()
    require_watertight(mesh, name)
    return {'name': name, 'base': base, 'mesh': mesh, 'tf': tf, 'visible': visible,
            'applied': applied, 'info': None}


def mesh_info(mesh):
    vol = abs(float(mesh.volume)) if mesh.is_watertight else 0.0
    return {'bounds': mesh.bounds.tolist(), 'extents': mesh.extents.tolist(),
            'watertight': bool(mesh.is_watertight), 'volume_mm3': vol,
            'volume_ml': vol / 1000.0}


def model_info(m):
    if m['info'] is None:          # cached; cleared whenever m['mesh'] changes
        m['info'] = mesh_info(m['mesh'])
    return m['info']


def rightmost_x():
    return max((float(m['mesh'].bounds[1, 0]) for m in S.models), default=None)


def rebuild_source():
    """Re-combine the visible models. Invalidates any generated mould and any
    generation that is still running."""
    visible = [m['mesh'].copy() for m in S.models if m['visible']]
    if visible:
        combined = trimesh.util.concatenate(visible)
        combined.remove_unreferenced_vertices()
        combined.fix_normals()
        S.generator.source = combined
    else:
        S.generator.source = None
    S.generator.parts = []
    S.generator.plug = None
    S.generated = False
    S.rev += 1
    if JOB is not None and JOB.state == 'running':
        JOB.cancel = True
        JOB.cancel_reason = 'changed'


def mesh_bytes(mesh, export=False):
    m = transform_to_export(mesh) if export else transform_to_preview(mesh)
    return m.export(file_type='stl')


def capacity_of(gen):
    return {'gap_ml': gen.cavity_volume_mm3 / 1000,
            'channel_ml': gen.pour_channel_volume_mm3 / 1000,
            'total_ml': gen.total_resin_volume_mm3 / 1000,
            'mix_ml': gen.total_resin_volume_mm3 * 1.1 / 1000}


def part_size_mm(mesh):
    """Printer-orientation size (X, Y, Z) of a part. The engine is Y-up."""
    e = mesh.extents
    return [float(e[0]), float(e[2]), float(e[1])]


# -------------------------------------------------------------------- routes
@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/status')
@locked
def status():
    models = []
    for i, m in enumerate(S.models):
        models.append({'index': i, 'name': m['name'], 'visible': m['visible'],
                       'tf': m['tf'], **model_info(m)})
    src = S.generator.source
    parts = []
    if S.generated:
        parts = [{'index': i, 'name': p.name, 'direction': list(p.direction),
                  'size_mm': part_size_mm(p.mesh),
                  'volume_mm3': abs(float(p.mesh.volume)) if p.mesh.is_watertight else 0.0}
                 for i, p in enumerate(S.generator.parts)]
    return jsonify({'ok': True, 'models': models, 'parts': parts, 'has_source': src is not None,
                    'generated': S.generated, 'source': mesh_info(src) if src is not None else None,
                    'flat': (S.generator.flat_stats if S.generated else None),
                    'capacity': capacity_of(S.generator)})


@app.post('/api/upload')
@locked
def upload():
    files = request.files.getlist('files')
    if not files:
        return err('Choose one or more STL, OBJ or 3MF files.')
    added = []
    added_models = []
    for f in files:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in MESH_EXTENSIONS:
            return err(f'{f.filename}: only STL, OBJ and 3MF files are supported.')
        data = f.read()
        try:
            mesh = trimesh.load(io.BytesIO(data), file_type=suffix.lstrip('.'), force='mesh')
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError('No triangle mesh found.')
            # Plain geometry-only mesh: drops UVs/colours/normals and merges vertices by
            # position so seams don't make the mesh look non-watertight.
            mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=True)
            # Reject non-watertight meshes immediately, before any processing.
            require_watertight(mesh, f.filename)
            # Z-up (STL/OBJ/3MF) -> engine Y-up.
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            mesh.apply_translation([-mesh.bounds.mean(axis=0)[0], -mesh.bounds[0, 1],
                                    -mesh.bounds.mean(axis=0)[2]])
            require_watertight(mesh, f.filename)
            existing = S.models + added_models
            if existing:
                existing_max_x = max(float(m['mesh'].bounds[1, 0]) for m in existing)
                mesh.apply_translation([existing_max_x + max(5.0, float(mesh.extents[0]) * .10)
                                        - float(mesh.bounds[0, 0]), 0, 0])
            added_models.append(make_model(Path(f.filename).name, mesh))
            added.append(Path(f.filename).name)
        except Exception as e:
            return err(f'{f.filename}: {e}')
    # All-or-nothing: only commit once every file has passed.
    S.models.extend(added_models)
    rebuild_source()
    return jsonify({'ok': True, 'added': added})


@app.post('/api/model/<int:index>/transform')
@locked
def transform(index):
    if index >= len(S.models):
        return err('Model not found.', 404)
    d = request.get_json(force=True)
    try:
        tf = clean_tf(d)
        mesh = apply_tf(S.models[index]['base'], tf)
        require_watertight(mesh, 'The transformed STL')
    except ValueError as e:
        return err(e)
    S.models[index]['tf'] = tf
    S.models[index]['applied'] = True
    S.models[index]['mesh'] = mesh
    S.models[index]['info'] = None
    rebuild_source()
    return jsonify({'ok': True, 'tf': tf})


@app.post('/api/model/<int:index>/visibility')
@locked
def visibility(index):
    if index >= len(S.models):
        return err('Model not found.', 404)
    S.models[index]['visible'] = bool((request.get_json(silent=True) or {}).get('visible', True))
    rebuild_source()
    return jsonify({'ok': True})


@app.post('/api/model/<int:index>/duplicate')
@locked
def duplicate(index):
    if index >= len(S.models):
        return err('Model not found.', 404)
    original = S.models[index]['mesh'].copy()
    cp = original.copy()
    cp.apply_translation([float(original.extents[0]) + max(5, float(original.extents[0]) * .10), 0, 0])
    # As before: the copy is the current (transformed) mesh, shifted along X, with a fresh identity transform.
    S.models.append(make_model(S.models[index]['name'] + ' (Copy)', cp))
    rebuild_source()
    return jsonify({'ok': True})


@app.delete('/api/model/<int:index>')
@locked
def delete_model(index):
    if index >= len(S.models):
        return err('Model not found.', 404)
    S.models.pop(index)
    rebuild_source()
    return jsonify({'ok': True})


@app.delete('/api/models')
@locked
def delete_all_models():
    S.models.clear()
    rebuild_source()
    return jsonify({'ok': True})


@app.post('/api/models/show-all')
@locked
def show_all():
    for m in S.models:
        m['visible'] = True
    rebuild_source()
    return jsonify({'ok': True})


# ------------------------------------------------------------- generate (job)
class JobCancelled(Exception):
    pass


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:8]
        self.state = 'running'        # running | done | error | cancelled
        self.message = 'Starting…'
        self.fraction = 0.0
        self.cancel = False
        self.cancel_reason = 'user'
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None


JOB = None


def parse_params(d):
    style = d.get('style', 'multi')
    if style not in ('multi', 'flat'):
        raise ValueError('Unknown mould type.')
    return {
        'style': style,
        'core': bool(d.get('core', False)),
        'panels': int(d.get('panels', 4)),
        'top_extra': float(d.get('top_extra', 8)),
        'wall': float(d.get('wall', 8)),
        'base': float(d.get('base', 6)),
        'clearance': float(d.get('clearance', 3)),
        'fill_diameter': float(d.get('fill_diameter', 12)),
        'fill_depth_percent': float(d.get('fill_depth_percent', 50)),
        'rim_width': float(d.get('rim_width', 2.5)),
        'rim_thickness': float(d.get('rim_thickness', 1.5)),
        'wall_style': d.get('wall_style', 'offset') if d.get('wall_style', 'offset') in ('offset', 'radial') else 'offset',
        'margin': float(d.get('margin', 10)),
        'label_tab': bool(d.get('label_tab', True)),
    }


def run_job(job, source, p, rev):
    def progress(message, fraction=None):
        if job.cancel:
            raise JobCancelled()
        job.message = message
        if fraction is not None:
            job.fraction = max(job.fraction, min(1.0, float(fraction)))

    try:
        gen = MoldGenerator()
        gen.source = source
        require_watertight(source, 'Source model')
        if p['style'] == 'flat':
            if p['core']:
                gen.warnings.append('Core: not available for 2D flat pour moulds, so it was skipped.')
            gen.generate_flat_2d(wall=p['wall'], base=p['base'], top_extra=p['top_extra'],
                                 margin=p['margin'], label_tab=p['label_tab'], progress=progress)
        else:
            gen.generate(panels=p['panels'], top_extra=p['top_extra'], wall=p['wall'],
                         base=p['base'], clearance=p['clearance'],
                         fill_diameter=p['fill_diameter'],
                         fill_depth_percent=p['fill_depth_percent'], core=p['core'],
                         rim_width=p['rim_width'], rim_thickness=p['rim_thickness'],
                         wall_style=p['wall_style'], progress=progress)
        if job.cancel:
            raise JobCancelled()
        with LOCK:
            if S.rev != rev:
                raise JobCancelled()
            S.generator = gen
            S.generated = True
        job.result = {'parts': [x.name for x in gen.parts], 'warnings': list(gen.warnings),
                      'capacity': capacity_of(gen)}
        job.fraction = 1.0
        job.message = 'Done.'
        job.state = 'done'
    except JobCancelled:
        job.state = 'cancelled'
        job.message = ('Cancelled: the models changed while the mould was being built.'
                       if job.cancel_reason == 'changed' else 'Cancelled.')
    except ValueError as e:
        job.state = 'error'
        job.error = str(e)
    except Exception as e:
        traceback.print_exc()
        job.state = 'error'
        job.error = str(e)
    finally:
        job.finished = time.time()


@app.post('/api/generate')
@locked
def generate():
    global JOB
    if JOB is not None and JOB.state == 'running':
        return err('A mould is already being generated.', 409)
    if S.generator.source is None:
        return err('Load and show at least one STL first.')
    try:
        params = parse_params(request.get_json(force=True))
        source = S.generator.source.copy()
        require_watertight(source, 'Source model')
    except (ValueError, TypeError) as e:
        return err(e)
    JOB = Job()
    threading.Thread(target=run_job, args=(JOB, source, params, S.rev), daemon=True).start()
    return jsonify({'ok': True, 'job': JOB.id})


@app.post('/api/generate/cancel')
def generate_cancel():
    job = JOB
    if job is not None and job.state == 'running':
        job.cancel = True
        job.cancel_reason = 'user'
    return jsonify({'ok': True})


@app.get('/api/job')
def job_status():
    job = JOB
    if job is None:
        return jsonify({'ok': True, 'state': 'idle'})
    end = job.finished or time.time()
    out = {'ok': True, 'state': job.state, 'id': job.id, 'message': job.message,
           'fraction': job.fraction, 'elapsed': round(end - job.started, 1)}
    if job.state == 'error':
        out['error'] = job.error
    if job.state == 'done':
        out.update(job.result)
    return jsonify(out)


# ------------------------------------------------------------------- preview
@app.get('/api/preview/source')
@locked
def preview_source():
    if S.generator.source is None:
        abort(404)
    # Before generation, show the imported source STL. Once generated, show the
    # exact plug mesh produced by the geometry engine, which already includes its
    # final base-relative placement. The client must not offset it again.
    preview_mesh = (S.generator.plug.copy()
                    if S.generated and S.generator.plug is not None
                    else S.generator.source.copy())
    return send_file(io.BytesIO(mesh_bytes(preview_mesh)), mimetype='model/stl',
                     download_name='Plug_Preview.stl' if S.generated else 'Source_Preview.stl')


@app.get('/api/preview/part/<int:index>')
@locked
def preview_part(index):
    if not S.generated or index >= len(S.generator.parts):
        abort(404)
    return send_file(io.BytesIO(mesh_bytes(S.generator.parts[index].mesh)),
                     mimetype='model/stl', download_name='Part_Preview.stl')


# -------------------------------------------------------------------- export
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
@locked
def export_part(index):
    if not S.generated or index >= len(S.generator.parts):
        abort(404)
    part = S.generator.parts[index]
    mesh = base_and_source_mesh() if part.name == 'Mold_Base' else part.mesh.copy()
    filename = 'Mold_Base_and_STL.stl' if part.name == 'Mold_Base' else part.name + '.stl'
    return send_file(io.BytesIO(mesh_bytes(mesh, True)), mimetype='application/sla',
                     as_attachment=True, download_name=filename)


@app.get('/api/export/plug')
@locked
def export_plug():
    if not S.generated or S.generator.plug is None:
        abort(404)
    return send_file(io.BytesIO(mesh_bytes(base_and_source_mesh(), True)),
                     mimetype='application/sla', as_attachment=True,
                     download_name='Mold_Base_and_STL.stl')


@app.get('/api/export/zip')
@locked
def export_zip():
    if not S.generated:
        abort(404)
    try:
        amount = max(0.0, min(100.0, float(request.args.get('explode', 0))))
    except ValueError:
        amount = 0.0
    max_size = max(S.generator.source.extents)
    dist = (amount / 100) * (8 + max(30, max_size * .30)) * 1.25
    items = []
    for p in S.generator.parts:
        if p.name == 'Mold_Base':
            mesh, filename = base_and_source_mesh(), 'Mold_Base_and_STL.stl'
        else:
            mesh, filename = p.mesh.copy(), p.name + '.stl'
        if amount:
            mesh.apply_translation(np.array(p.direction) * dist)   # zero for the base
        items.append((filename, mesh))
    if amount:
        # Exploded: one shared rebase keeps the parts apart. (Rebasing each part
        # to Z=0 on its own would drop a lifted core back into the base.)
        exported = transform_set_to_export([m for _, m in items])
    else:
        # Normal: each part sits flat on Z=0, ready to print.
        exported = [transform_to_export(m) for _, m in items]
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, 'w', zipfile.ZIP_DEFLATED) as zf:
        for (filename, _), mesh in zip(items, exported):
            zf.writestr(filename, mesh.export(file_type='stl'))
    buff.seek(0)
    return send_file(buff, mimetype='application/zip', as_attachment=True,
                     download_name='Mad_Hat_Mold_Parts.zip')


# ------------------------------------------------------------------- project
@app.post('/api/project')
@locked
def project_save():
    """Project archive (version 2): each model's original laid-out mesh, its
    transform and visibility, plus the mould settings sent by the client."""
    body = request.get_json(silent=True) or {}
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, 'w', zipfile.ZIP_DEFLATED) as z:
        meta = []
        for i, m in enumerate(S.models):
            z.writestr(f'models/model_{i + 1:03d}.stl', m['base'].export(file_type='stl'))
            meta.append({'name': m['name'], 'visible': m['visible'], 'tf': m['tf'],
                         'applied': m['applied']})
        z.writestr('project.json', json.dumps(
            {'version': 2, 'models': meta, 'settings': body.get('settings', {})}, indent=2))
    buff.seek(0)
    return send_file(buff, mimetype='application/zip', as_attachment=True,
                     download_name='Mad_Hat_Project.zip')


@app.post('/api/project/open')
@locked
def project_open():
    f = request.files.get('file')
    if not f:
        return err('Choose a project ZIP.')
    try:
        z = zipfile.ZipFile(io.BytesIO(f.read()))
        if sum(i.file_size for i in z.infolist()) > MAX_PROJECT_UNCOMPRESSED:
            raise ValueError('The archive is too large to open safely.')
        meta = json.loads(z.read('project.json'))
        models = []
        for i, item in enumerate(meta['models']):
            data = z.read(f'models/model_{i + 1:03d}.stl')
            mesh = trimesh.load(io.BytesIO(data), file_type='stl', force='mesh')
            mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=True)
            # Project models are stored in the engine's native Y-up coordinates.
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            require_watertight(mesh, item['name'])
            # Version 1 projects stored the already-transformed mesh and no tf;
            # they load as identity transforms.
            tf = clean_tf(item.get('tf'))
            models.append(make_model(item['name'], mesh, tf, bool(item.get('visible', True)),
                                     bool(item.get('applied', not is_identity(tf)))))
        S.models = models
        rebuild_source()
        return jsonify({'ok': True, 'settings': meta.get('settings', {})})
    except Exception as e:
        return err(f'Could not open project: {e}')


if __name__ == '__main__':
    print('\nMAD HAT MOULDFORGE — Web Interface')
    print('Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.\n')
    if vendor.missing():
        print('Downloading the 3D viewer files once so the app can run offline...\n')
    vendor.ensure_async()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
