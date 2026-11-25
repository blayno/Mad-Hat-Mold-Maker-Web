"""
MAD HAT — Negative Mold Generator (geometry engine)

Install:
    pip install flask trimesh manifold3d numpy scipy shapely scikit-image

The engine works Y-up internally. Imported meshes are Z-up and are rotated on
import (see app.py); generated STL exports are Z-up again.
Boolean operations use the Manifold3D engine via trimesh.

progress(message, fraction=None) callbacks are optional. A callback may raise
to abort generation (app.py uses this for Cancel).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh


GOLD = "#f0a500"
PALETTE = ["#e34b91", "#9b58e8", "#d5ae45", "#73c95d",
           "#39b8a9", "#8793a8", "#e07848", "#6b9fe0"]


def box_mesh(extents, center):
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def vertical_cylinder(radius, height, sections=32):
    """Cylinder whose axis is the application's Y-up vertical axis."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2, [1,0,0]))
    return mesh


def cylinder_between(radius, depth, start, direction, sections=32):
    """Cylinder centred halfway along a vector."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    mid = np.asarray(start, dtype=float) + direction * depth / 2
    mesh = trimesh.creation.cylinder(radius=radius, height=depth, sections=sections)
    T = trimesh.geometry.align_vectors([0, 0, 1], direction)
    if T is not None:
        mesh.apply_transform(T)
    mesh.apply_translation(mid)
    return mesh


def boolean(meshes, operation):
    if not meshes:
        raise ValueError("No meshes supplied to Boolean operation.")
    if operation == "union":
        result = trimesh.boolean.union(meshes, engine="manifold")
    elif operation == "difference":
        result = trimesh.boolean.difference(meshes, engine="manifold")
    elif operation == "intersection":
        result = trimesh.boolean.intersection(meshes, engine="manifold")
    else:
        raise ValueError(operation)
    if result is None:
        raise RuntimeError("Manifold Boolean failed. Check that the STL is watertight.")
    if isinstance(result, trimesh.Scene):
        result = trimesh.util.concatenate(tuple(result.geometry.values()))
    result.remove_unreferenced_vertices()
    return result


# --- Label tab (2D flat mould) ----------------------------------------------
# A tiny 5x7 pixel font (digits, '.', 'M', 'L', space) so the mould can carry
# its silicone amount as raised text without needing any font files.
_PIXEL_FONT = {
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    ".": ["..", "..", "..", "..", "..", "##", "##"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    " ": ["...", "...", "...", "...", "...", "...", "..."],
}


def pixel_text_rects(text):
    """Lay text out on a pixel grid.

    Returns (rects, cols, rows). Each rect is (col0, col1, row0, row1) in grid
    units (end-exclusive, row 0 = top row), with vertical runs already merged
    so the text is built from as few boxes as possible.
    """
    runs = []
    col = 0
    for ch in text:
        glyph = _PIXEL_FONT.get(ch)
        if glyph is None:
            continue
        for r, line in enumerate(glyph):
            c = 0
            while c < len(line):
                if line[c] == "#":
                    c0 = c
                    while c < len(line) and line[c] == "#":
                        c += 1
                    runs.append((col + c0, col + c, r))
                else:
                    c += 1
        col += len(glyph[0]) + 1
    rects = []
    for c0, c1 in sorted({(a, b) for a, b, _ in runs}):
        rows = sorted(r for a, b, r in runs if (a, b) == (c0, c1))
        start = prev = rows[0]
        for r in rows[1:]:
            if r != prev + 1:
                rects.append((c0, c1, start, prev + 1))
                start = r
            prev = r
        rects.append((c0, c1, start, prev + 1))
    return rects, max(0, col - 1), 7


def build_label_tab(outer_shape, wall, base_top, text, text_height=1.0, pad=2.0):
    """Flat tab on the top edge of the 2D mould with raised text on it.

    "Top edge" is the top of the top-down view, i.e. the smallest engine Z
    (the engine is Y-up and the preview/export rotate +90 deg about X, so text
    laid out along +X with rows increasing in +Z reads correctly from above).
    The tab is a floor-level plate that overlaps into the wall ring but never
    enters the silicone well. Returns a list of meshes to union with the tray.
    """
    from shapely.geometry import LineString

    rects, cols, rows = pixel_text_rects(text)
    if not rects:
        return []
    minx, minz, maxx, maxz = outer_shape.bounds
    # Shrink the pixel size on small moulds, but keep strokes printable.
    avail = max(30.0, maxx - minx)
    pitch = min(1.2, max(0.6, (avail - 2.0 * pad) / cols))
    text_w, text_d = cols * pitch, rows * pitch
    tab_w, tab_d = text_w + 2.0 * pad, text_d + 2.0 * pad

    # Centre the tab on the outline vertices nearest the top edge, then find
    # where the outline actually is at that X so the plate overlaps the wall.
    coords = np.asarray(outer_shape.exterior.coords)
    near_top = coords[coords[:, 1] <= minz + 1.0]
    xc = float(near_top[:, 0].mean())
    hit = LineString([(xc, minz - 1.0), (xc, maxz + 1.0)]).intersection(outer_shape)
    if hit.is_empty:
        xc = float(coords[np.argmin(coords[:, 1]), 0])
        hit = LineString([(xc, minz - 1.0), (xc, maxz + 1.0)]).intersection(outer_shape)
    z_loc = float(hit.bounds[1]) if not hit.is_empty else float(minz)

    embed = wall * 0.5  # sink into the wall ring (ring is at least `wall` thick)
    z_out = z_loc - tab_d
    meshes = [box_mesh([tab_w, base_top, tab_d + embed],
                       [xc, base_top / 2.0, (z_out + z_loc + embed) / 2.0])]
    x0 = xc - text_w / 2.0
    z0 = z_out + pad
    h = text_height + 0.2  # sink 0.2 mm into the plate so it fuses cleanly
    for c0, c1, r0, r1 in rects:
        meshes.append(box_mesh(
            [(c1 - c0) * pitch, h, (r1 - r0) * pitch],
            [x0 + (c0 + c1) * pitch / 2.0,
             base_top - 0.2 + h / 2.0,
             z0 + (r0 + r1) * pitch / 2.0]))
    return meshes


def _dense_surface_points(mesh, row_height, max_faces=600_000):
    """Points covering the whole surface, not just the mesh vertices.

    Meshes from 3MF/OBJ (and decimated STLs) often have long thin triangles, so
    their vertices alone leave many radial bins empty.  The empty bins were then
    filled from neighbours or a fallback radius, which produced the spiky base
    and stepped cavity.  Subdividing the triangles first gives every bin real
    surface samples.  The subdivided points still lie on the original surface.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    try:
        edge = min(1.5, max(0.3, float(row_height) * 0.5))
        # Roughly 0.43 * edge^2 of area per triangle; keep the face count sane.
        est_faces = float(mesh.area) / (0.43 * edge * edge)
        if est_faces > max_faces:
            edge *= math.sqrt(est_faces / max_faces)
        dense = mesh.subdivide_to_size(max_edge=edge, max_iter=10)
        return np.vstack([verts, np.asarray(dense.vertices, dtype=float)])
    except Exception:
        try:
            n = int(min(400_000, max(50_000, float(mesh.area) / 0.09)))
            return np.vstack([verts, np.asarray(mesh.sample(n), dtype=float)])
        except Exception:
            return verts


def radial_profile(mesh, height, angle_count=144, y_count=48):
    """Conservative radial envelope of the model at sampled heights/angles."""
    v = _dense_surface_points(mesh, height / max(y_count, 1))
    radii = np.full((y_count + 1, angle_count), -np.inf)
    yi = np.clip(np.rint(v[:, 1] / max(height, 1e-8) * y_count).astype(int), 0, y_count)
    angles = (np.arctan2(v[:, 0], v[:, 2]) + 2 * np.pi) % (2 * np.pi)
    ai = np.floor(angles / (2 * np.pi) * angle_count).astype(int) % angle_count
    rr = np.hypot(v[:, 0], v[:, 2])
    np.maximum.at(radii, (yi, ai), rr)
    radii[~np.isfinite(radii)] = np.nan

    # Fill sparse bins using a conservative neighbourhood maximum.
    for _ in range(3):
        old = radii.copy()
        for y in range(y_count + 1):
            for a in range(angle_count):
                if np.isfinite(old[y, a]):
                    continue
                vals = []
                for dy in range(-3, 4):
                    yy = y + dy
                    if 0 <= yy <= y_count:
                        for da in range(-4, 5):
                            val = old[yy, (a + da) % angle_count]
                            if np.isfinite(val):
                                vals.append(val)
                if vals:
                    radii[y, a] = max(vals)

    fallback = max(mesh.extents[0], mesh.extents[2]) / 2
    radii[~np.isfinite(radii)] = fallback
    # Conservative smoothing; keep the envelope outside measured vertices.
    smooth = radii.copy()
    for y in range(1, y_count):
        for a in range(angle_count):
            avg = (radii[y, a] * 4 + radii[y, (a-1) % angle_count] +
                   radii[y, (a+1) % angle_count] + radii[y-1, a] +
                   radii[y+1, a]) / 8
            smooth[y, a] = max(radii[y, a], avg)
    pad = max(1.0, min(2.0, max(mesh.extents[0], mesh.extents[2]) * .004))
    smooth += pad

    # Smooth only the outer wall profile.
    outer = smooth.copy()
    kernel = np.array([1, 4, 6, 4, 1], dtype=float) / 16
    for _ in range(8):
        tmp = np.empty_like(outer)
        for y in range(y_count + 1):
            tmp[y] = sum(kernel[k+2] * np.roll(outer[y], k) for k in range(-2, 3))
        new = np.empty_like(tmp)
        for y in range(y_count + 1):
            new[y] = sum(kernel[k+2] * tmp[np.clip(y+k, 0, y_count)] for k in range(-2, 3))
        outer = new
    first = int(math.ceil(.90 * y_count))
    opening = float(np.max(smooth[first]))
    smooth[first:] = opening
    return {"inner": smooth, "outer": outer, "height": height,
            "angles": angle_count, "rows": y_count}


def profile_radius(profile, angle, y, outer=False):
    rows = profile["outer"] if outer else profile["inner"]
    n, ny = profile["angles"], profile["rows"]
    af = ((angle % (2*np.pi)) / (2*np.pi)) * n
    a0 = int(math.floor(af)) % n
    a1 = (a0 + 1) % n
    at = af - math.floor(af)
    yf = np.clip(y / max(profile["height"], 1e-8) * ny, 0, ny)
    y0 = int(math.floor(yf))
    y1 = min(ny, y0 + 1)
    yt = yf - y0
    r0 = rows[y0, a0] * (1-at) + rows[y0, a1] * at
    r1 = rows[y1, a0] * (1-at) + rows[y1, a1] * at
    return float(r0 * (1-yt) + r1 * yt)


def offset_model(mesh, clearance, return_fields=False, keep_sdf=False):
    """Build a robust outward offset from a filled voxel distance field.

    Vertex-normal offsets can fold over on dense, highly curved meshes, so the
    source is voxelised, filled, and a Euclidean distance transform gives the
    distance from every voxel to the solid.  The cutter surface is extracted
    with marching cubes on that *distance field* at the clearance level, which
    places vertices with sub-voxel accuracy.  (Marching cubes on a binary mask
    would put every vertex on a voxel face and leave visible stair-stepping.)

    The voxel pitch is capped at 0.5 mm for useful detail without creating an
    excessive memory footprint; very large models get a coarser pitch, which
    is reported in fields["pitch"].
    """
    from scipy import ndimage
    from skimage import measure

    distance = max(0.0, float(clearance)) + 0.15
    if not np.isfinite(distance):
        raise ValueError("Cavity clearance must be a finite number.")

    extents = np.asarray(mesh.extents, dtype=float)
    if np.any(~np.isfinite(extents)) or np.any(extents <= 0):
        raise ValueError("Cannot offset a mesh with invalid dimensions.")

    pitch = min(0.5, max(0.35, distance / 6.0))
    estimated_voxels = float(np.prod(np.ceil(extents / pitch + 4)))
    if estimated_voxels > 14_000_000:
        pitch *= (estimated_voxels / 14_000_000) ** (1.0 / 3.0)

    try:
        surface = mesh.voxelized(pitch=pitch)
        solid = surface.fill()
        occupied = np.asarray(solid.matrix, dtype=bool)
        if occupied.size == 0 or not occupied.any():
            raise ValueError("Voxelisation produced an empty model.")

        # Empty space around the source for the complete offset.
        pad = int(math.ceil(distance / pitch)) + 3
        padded = np.pad(occupied, pad, mode="constant", constant_values=False)

        # Signed distance to the filled volume, measured from the voxel
        # surface (half a voxel from the occupied voxel centres): positive
        # outside, negative inside.
        outside = ndimage.distance_transform_edt(~padded, sampling=pitch)
        inside = ndimage.distance_transform_edt(padded, sampling=pitch)
        sdf = np.where(padded, -(inside - 0.5 * pitch), outside - 0.5 * pitch)
        sdf = sdf.astype(np.float32)
        # Very light smoothing removes the residual EDT ripple. At sigma 0.5
        # voxels the surface moves by far less than the clearance.
        sdf = ndimage.gaussian_filter(sdf, sigma=0.5)

        offset_volume = sdf <= distance          # source + clearance (binary)

        grid_origin = np.asarray(surface.transform[:3, 3], dtype=float)
        origin = grid_origin - pad * pitch       # world position of voxel [0,0,0]
        verts, faces, _normals, _values = measure.marching_cubes(
            sdf, level=float(distance), spacing=(pitch, pitch, pitch))
        out = trimesh.Trimesh(vertices=verts + origin, faces=faces, process=True)
        out.remove_unreferenced_vertices()
        out.fix_normals()

        if (not out.is_watertight or not out.is_winding_consistent or
                not np.isfinite(out.vertices).all() or out.volume <= 0):
            raise RuntimeError(
                "The volumetric cavity offset did not produce a valid closed solid."
            )
        if return_fields:
            fields = {
                "padded": padded,                  # filled source voxels
                "offset_volume": offset_volume,    # source + clearance
                "pitch": float(pitch),
                "origin": origin,
                "sdf": sdf if keep_sdf else None,
            }
            return out, fields
        return out
    except Exception as exc:
        raise RuntimeError(
            f"Could not create a clean {distance:.2f} mm cavity offset: {exc}"
        ) from exc


def profile_radius_grid(profile, angle, y, outer=False):
    """Vectorised profile_radius: `angle` and `y` are broadcast against each other."""
    rows = profile["outer"] if outer else profile["inner"]
    n, ny = profile["angles"], profile["rows"]
    angle = np.asarray(angle, dtype=float)
    y = np.asarray(y, dtype=float)
    af = ((angle % (2 * np.pi)) / (2 * np.pi)) * n
    a_floor = np.floor(af)
    at = af - a_floor
    a0 = a_floor.astype(int) % n
    a1 = (a0 + 1) % n
    yf = np.clip(y / max(profile["height"], 1e-8) * ny, 0, ny)
    y_floor = np.floor(yf)
    yt = yf - y_floor
    y0 = y_floor.astype(int)
    y1 = np.minimum(ny, y0 + 1)
    r0 = rows[y0, a0] * (1 - at) + rows[y0, a1] * at
    r1 = rows[y1, a0] * (1 - at) + rows[y1, a1] * at
    return r0 * (1 - yt) + r1 * yt


def annular_panel_arrays(profile, a0, a1, y_min, y_max, wall, clearance, base_top,
                         na=192, ny=128):
    """Vertices/faces of one annular wall panel blank (vectorised)."""
    reserve = max(5.0, wall * .55)
    ang = a0 + (a1 - a0) * np.arange(na + 1) / na            # (na+1,)
    ys = y_min + (y_max - y_min) * np.arange(ny + 1) / ny    # (ny+1,)
    A, Y = np.meshgrid(ang, ys)                              # (ny+1, na+1)
    source_y = Y - base_top
    base_r = profile_radius_grid(profile, A, np.zeros_like(A)) + clearance
    ri = np.where(Y <= base_top + 1e-6, base_r, .25)
    ro = profile_radius_grid(profile, A, source_y, True) + clearance + wall + reserve
    verts = []
    for r in (ri, ro):
        rr = np.maximum(.2, r)
        verts.append(np.stack([np.sin(A) * rr, Y, np.cos(A) * rr], axis=-1).reshape(-1, 3))
    verts = np.vstack(verts)

    grid = (na + 1) * (ny + 1)
    faces = []

    def quad(a, b, c, d):
        faces.append(np.stack([a, b, d], axis=-1))
        faces.append(np.stack([b, c, d], axis=-1))

    jj, ii = np.meshgrid(np.arange(ny), np.arange(na), indexing="ij")
    jj, ii = jj.ravel(), ii.ravel()

    def idx(layer, i, j):
        return layer * grid + j * (na + 1) + i

    for layer in range(2):
        a, b = idx(layer, ii, jj), idx(layer, ii + 1, jj)
        c, d = idx(layer, ii + 1, jj + 1), idx(layer, ii, jj + 1)
        if layer == 0:
            quad(a, d, c, b)
        else:
            quad(a, b, c, d)
    j = np.arange(ny)
    quad(idx(0, 0, j), idx(1, 0, j), idx(1, 0, j + 1), idx(0, 0, j + 1))
    quad(idx(0, na, j), idx(0, na, j + 1), idx(1, na, j + 1), idx(1, na, j))
    i = np.arange(na)
    quad(idx(0, i, 0), idx(1, i, 0), idx(1, i + 1, 0), idx(0, i + 1, 0))
    quad(idx(0, i, ny), idx(1, i, ny), idx(1, i + 1, ny), idx(0, i + 1, ny))
    return verts, np.vstack(faces)


def annular_panel(profile, a0, a1, y_min, y_max, wall, clearance, base_top):
    verts, faces = annular_panel_arrays(profile, a0, a1, y_min, y_max, wall,
                                        clearance, base_top)
    result = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    result.fix_normals()
    return result


def contoured_clip_flange(profile, angle, y_min, y_max, base_top,
                          clearance=0.0, wall=8.0, projection=12.0,
                          thickness=3.0, tangential_offset=0.0,
                          segments=64):
    """
    Build the same height-sampled, contour-following clip flange as the HTML
    generator. The flange root penetrates the panel's outer skin so the union
    is reliable, while its outside face follows the smoothed wall profile.
    """
    verts = []
    faces = []
    tangent_min = tangential_offset - thickness / 2
    tangent_max = tangential_offset + thickness / 2
    outer_wall_reserve = max(5.0, wall * 0.55)
    root_overlap = float(np.clip(wall * 0.2, 1.5, 3.0))

    for j in range(segments + 1):
        y = y_min + (y_max - y_min) * j / segments
        source_y = float(np.clip(y - base_top, 0, profile["height"]))
        panel_outer = (profile_radius(profile, angle, source_y, True)
                       + max(0.0, clearance) + wall + outer_wall_reserve)
        inner_radius = panel_outer - root_overlap
        outer_radius = panel_outer + max(0.0, projection)
        for radius, tangent in (
            (inner_radius, tangent_min),
            (outer_radius, tangent_min),
            (outer_radius, tangent_max),
            (inner_radius, tangent_max),
        ):
            verts.append([
                math.sin(angle) * radius + math.cos(angle) * tangent,
                y,
                math.cos(angle) * radius - math.sin(angle) * tangent,
            ])

    def ring(j, k):
        return j * 4 + k

    for j in range(segments):
        for k in range(4):
            nxt = (k + 1) % 4
            a, b = ring(j, k), ring(j, nxt)
            c, d = ring(j + 1, nxt), ring(j + 1, k)
            faces.extend([[a, b, d], [b, c, d]])

    faces.extend([
        [ring(0, 0), ring(0, 2), ring(0, 1)],
        [ring(0, 0), ring(0, 3), ring(0, 2)],
    ])
    top = ring(segments, 0)
    faces.extend([[top, top + 1, top + 2], [top, top + 2, top + 3]])

    flange = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )
    flange.fix_normals()
    return flange


def flange_seal_ring(angle, profile, y_min, y_max, base_top,
                     clearance=0.0, wall=8.0, projection=12.0,
                     tangential_face=0.0, rim_width=2.5, rim_thickness=1.5,
                     grow=0.0, segments=64, sides=12,
                     line_y_min=None, line_y_max=None):
    """Create one rounded sealing bead along the flange centreline.

    The bead runs vertically between the two locating-pin/socket centres,
    following the changing radial contour of the flange.  It is deliberately
    a single line, not a closed perimeter loop.
    """
    cord_radius = max(0.15, (float(rim_thickness) + 2.0 * grow) / 2.0)
    y0 = float(y_min if line_y_min is None else line_y_min)
    y1 = float(y_max if line_y_max is None else line_y_max)
    if y1 <= y0:
        return None

    ys = np.linspace(y0, y1, max(8, int(segments)) + 1)
    outer_wall_reserve = max(5.0, wall * 0.55)
    # Place the single bead halfway across the flange projection, between
    # the inner flange root and its outer edge.
    radii = []
    for y in ys:
        source_y = float(np.clip(y - base_top, 0, profile["height"]))
        panel_outer = (
            profile_radius(profile, angle, source_y, True)
            + max(0.0, clearance) + wall + outer_wall_reserve
        )
        radii.append(panel_outer + max(0.0, projection) * 0.5)

    path = np.column_stack([np.asarray(radii, dtype=float), ys])
    verts, faces = [], []
    for i, (r, y) in enumerate(path):
        if i == 0:
            direction = path[1] - path[0]
        elif i == len(path) - 1:
            direction = path[-1] - path[-2]
        else:
            direction = path[i + 1] - path[i - 1]
        length = float(np.linalg.norm(direction))
        if length < 1e-10:
            direction = np.array([0.0, 1.0])
            length = 1.0
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        for j in range(sides):
            a = 2.0 * math.pi * j / sides
            rr = r + cord_radius * math.cos(a) * normal[0]
            yy = y + cord_radius * math.cos(a) * normal[1]
            tt = tangential_face + cord_radius * math.sin(a)
            verts.append([rr, yy, tt])

    for i in range(len(path) - 1):
        for j in range(sides):
            nj = (j + 1) % sides
            a = i * sides + j
            b = i * sides + nj
            c = (i + 1) * sides + nj
            d = (i + 1) * sides + j
            faces.extend([[a, b, d], [b, c, d]])
    # Close both ends of the bead with triangular fans.
    for j in range(1, sides - 1):
        faces.append([0, j + 1, j])
    last = (len(path) - 1) * sides
    for j in range(1, sides - 1):
        faces.append([last, last + j, last + j + 1])

    bead = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64), process=True)
    radial = np.array([math.sin(angle), 0.0, math.cos(angle)])
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    transform = np.eye(4)
    transform[:3, 0] = radial
    transform[:3, 1] = [0.0, 1.0, 0.0]
    transform[:3, 2] = tangent
    bead.apply_transform(transform)
    bead.fix_normals()
    return bead


# Clearance (mm) on every side between the base lip and the matching groove
# cut into the bottom of the mould walls.  Was 0.2 (too tight once printed).
BASE_RIM_GAP = 0.5


def base_tab_ring(profile, y_center, height, projection, clearance,
                  root=1.0, grow=0.0, segments=192):
    """Closed rectangular ring around the base's outer edge.

    Its root sinks `root` mm into the base disc and it sticks out `projection`
    mm past the disc edge, centred vertically on `y_center`.  With grow > 0 it
    is enlarged (used as the matching groove cutter for the wall panels).
    """
    h = height + 2 * grow
    y0 = y_center - h / 2
    verts = []
    for i in range(segments):
        ang = i / segments * 2 * np.pi
        edge = max(1.0, profile_radius(profile, ang, 0) + clearance - 0.2)   # base disc edge
        for r, y in ((edge - root, y0), (edge + projection + grow, y0),
                     (edge + projection + grow, y0 + h), (edge - root, y0 + h)):
            verts.append([math.sin(ang) * r, y, math.cos(ang) * r])
    faces = []
    for i in range(segments):
        n = (i + 1) % segments
        for k in range(4):
            nk = (k + 1) % 4
            a_, b_ = i * 4 + k, i * 4 + nk
            c_, d_ = n * 4 + nk, n * 4 + k
            faces.extend([[a_, b_, d_], [b_, c_, d_]])
    ring = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64), process=True)
    ring.fix_normals()
    return ring


def radial_prism(profile, y0, y1, clearance, subtract=0.2, count=192):
    verts, faces = [], []
    for y in (y0, y1):
        for i in range(count):
            a = i/count*2*np.pi
            r = max(1, profile_radius(profile, a, 0) + clearance - subtract)
            verts.append([math.sin(a)*r, y, math.cos(a)*r])
    bottom_center = len(verts); verts.append([0,y0,0])
    top_center = len(verts); verts.append([0,y1,0])
    for i in range(count):
        n=(i+1)%count
        faces.extend([[i,n,count+n],[i,count+n,count+i],
                      [bottom_center,n,i],[top_center,count+i,count+n]])
    out=trimesh.Trimesh(vertices=np.asarray(verts),faces=np.asarray(faces),process=True)
    out.fix_normals()
    return out



def _mask_to_mesh(mask, pitch, origin):
    """Marching-cubes a boolean voxel mask into a closed mesh (world coords)."""
    from skimage import measure
    vol = np.pad(mask, 1, mode="constant", constant_values=False).astype(np.float32)
    verts, faces, _n, _v = measure.marching_cubes(vol, level=0.5,
                                                  spacing=(pitch, pitch, pitch))
    verts = verts + np.asarray(origin, dtype=float) - pitch
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def core_voxels(fields, top_y, fill_diameter, cap_height=3.0):
    """Voxel-side core logic (no Boolean engine needed).

    Finds cavities that are enclosed sideways but open upwards (cup / bowl /
    vase interiors), shrinks them by the silicone clearance, extends them
    straight up to the mould top as a stem, and works out where the pour hole
    and the retaining cap go.  Returns a dict, or None plus warnings.
    """
    from scipy import ndimage

    padded = fields["padded"]
    offset = fields["offset_volume"]
    pitch = fields["pitch"]
    origin = np.asarray(fields["origin"], dtype=float)
    warnings = []

    # 1. Pockets: empty voxels that are enclosed within their own horizontal
    #    (X/Z) slice.  Connectivity is restricted to the slice plane so a cup's
    #    interior counts even though it is open at the top.
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[:, 1, :] = ndimage.generate_binary_structure(2, 1)
    filled = ndimage.binary_fill_holes(padded, structure=struct)
    pocket = filled & ~padded

    # The core is the pocket minus the silicone gap (the cutter's offset).
    raw = pocket & ~offset
    if not raw.any():
        warnings.append("Core: no open cavity found in the model, so no core was made.")
        return None, warnings

    # 2. Which voxels can see the sky (nothing of the cutter above them)?
    blocked_above = np.flip(np.cumsum(np.flip(offset, axis=1), axis=1, dtype=np.int32), axis=1)
    sky = raw & (blocked_above == 0)

    labels, n = ndimage.label(raw)
    sizes = np.bincount(labels[raw], minlength=n + 1)
    sky_counts = np.bincount(labels[sky], minlength=n + 1)
    candidates = [l for l in range(1, n + 1) if sky_counts[l] > 0 and sizes[l] > 0]
    if not candidates:
        warnings.append("Core: the cavity is not open at the top (rotate the model so the "
                        "opening faces up), so no core was made.")
        return None, warnings
    chosen = max(candidates, key=lambda l: sizes[l])
    if len(candidates) > 1:
        warnings.append(f"Core: {len(candidates)} separate open cavities found; only the "
                        "largest was cored.")
    core = labels == chosen
    undercut = 1.0 - sky_counts[chosen] / float(sizes[chosen])
    if undercut > 0.02:
        warnings.append(f"Core: about {undercut*100:.0f}% of the core sits under an overhang "
                        "(cavity wider than its opening). It will not pull out upwards in one piece.")

    # 3. Stem: extrude every unobstructed column straight up to the mould top.
    k_target = int(math.ceil((top_y + 0.5 - origin[1]) / pitch))
    if k_target >= core.shape[1]:
        core = np.pad(core, ((0, 0), (0, k_target + 1 - core.shape[1]), (0, 0)))
        blocked_above = np.pad(blocked_above, ((0, 0), (0, k_target + 1 - blocked_above.shape[1]), (0, 0)))
    ny = core.shape[1]
    has_core = core.any(axis=1)
    top_idx = ny - 1 - np.argmax(core[:, ::-1, :], axis=1)
    top_free = np.take_along_axis(blocked_above, top_idx[:, None, :], axis=1)[:, 0, :] == 0
    open_cols = has_core & top_free
    ys = np.arange(ny)[None, :, None]
    stem = open_cols[:, None, :] & (ys > top_idx[:, None, :]) & (ys <= k_target)
    core_full = core | stem

    stem2d = core_full[:, k_target, :]
    if not stem2d.any():
        warnings.append("Core: could not extend the core to the top of the mould.")
        return None, warnings

    # The removable core stays solid: do not bore a fill hole through its centre.
    # The silicone pour opening is cut through the mould panels separately.
    hole = None

    # 5. Cap centre / radius.
    ii, kk = np.nonzero(stem2d)
    cx = origin[0] + ii.mean() * pitch
    cz = origin[2] + kk.mean() * pitch
    spread = np.hypot(origin[0] + ii * pitch - cx, origin[2] + kk * pitch - cz).max()
    cap_r = float(spread + pitch + 6.0)

    # 6. Panel bore: the core dilated by one voxel (~0.35-0.5 mm) for a print fit.
    cut = ndimage.binary_dilation(core_full, structure=ndimage.generate_binary_structure(3, 1),
                                  iterations=1)

    return {"core": core_full, "cut": cut, "pitch": pitch, "origin": origin,
            "hole": hole, "cap": (float(cx), float(cz), cap_r), "cap_height": float(cap_height),
            "top_y": float(top_y)}, warnings


def build_core(fields, top_y, fill_diameter, cap_height=3.0):
    """Create the removable core meshes in the source's own coordinates.

    Returns (core_mesh, panel_cut_mesh, warnings); the meshes are None when no
    usable core could be made.
    """
    info, warnings = core_voxels(fields, top_y, fill_diameter, cap_height)
    if info is None:
        return None, None, warnings
    pitch, origin = info["pitch"], info["origin"]
    core = _mask_to_mesh(info["core"], pitch, origin)
    cut = _mask_to_mesh(info["cut"], pitch, origin)

    cx, cz, cap_r = info["cap"]
    cap = vertical_cylinder(cap_r, cap_height, sections=96)
    cap.apply_translation([cx, top_y + cap_height / 2, cz])
    core = boolean([core, cap], "union")

    core.fix_normals()
    return core, cut, warnings


# --- Uniform-thickness outer wall (offset envelope) -------------------------
def surface_from_sdf(fields, level):
    """Marching-cubes surface of an existing offset field at another distance.

    Reuses the signed distance field from offset_model(..., keep_sdf=True), so a
    deeper offset (e.g. the locating-key depth) costs no extra distance transform.
    """
    from skimage import measure
    sdf = fields.get("sdf")
    if sdf is None:
        raise RuntimeError("Offset field was built without its distance field.")
    pitch = float(fields["pitch"])
    verts, faces, _n, _v = measure.marching_cubes(
        sdf, level=float(level), spacing=(pitch, pitch, pitch))
    out = trimesh.Trimesh(vertices=verts + np.asarray(fields["origin"], dtype=float),
                          faces=faces, process=True)
    out.remove_unreferenced_vertices()
    out.fix_normals()
    if not out.is_watertight or out.volume <= 0:
        raise RuntimeError("The locating-key surface was not a closed solid.")
    return out


def key_slab(angle, y0, y1, radius, width, offset=0.0):
    """Box on the seam plane at `angle`: `width` wide across the seam, 0..radius out."""
    t = np.array([math.cos(angle), 0.0, -math.sin(angle)])      # across the seam
    rad = np.array([math.sin(angle), 0.0, math.cos(angle)])     # along the seam, outward
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = t, [0.0, 1.0, 0.0], rad
    normal = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    m[:3, 3] = rad * radius / 2.0 + normal * float(offset) + np.array([0.0, (y0 + y1) / 2.0, 0.0])
    return trimesh.creation.box(extents=[float(width), float(y1 - y0), float(radius)], transform=m)


def wall_thickness_estimate(fields, clearance, wall, y_top, hole_radius,
                            hole_y0, boss=None):
    """Estimate the thinnest part of the outer wall (mm) from the offset field.

    Uses the signed distance field from offset_model(..., keep_sdf=True) run at
    clearance + wall.  Every outer-surface voxel's distance to the cavity gives the
    wall thickness there, so the offset wall reads as `wall` and anywhere the top is
    cut flat below the offset surface reads thinner.  The pour hole adds a second
    limit (distance from the outer surface to the hole).  Voxel quantisation means
    the result is good to about +/- half a voxel (~0.25 mm).  Flanges, the rim
    groove and the core bore are not included.
    Returns {"min_mm", "y_src", "where", "pitch"} or None.
    """
    from scipy import ndimage
    sdf = fields.get("sdf")
    if sdf is None:
        return None
    pitch = float(fields["pitch"])
    origin = np.asarray(fields["origin"], dtype=float)
    nx, ny, nz = sdf.shape
    cav = float(clearance) + 0.15            # cavity surface distance used by the cutter
    env = np.asarray(fields["offset_volume"], dtype=bool).copy()
    xs = origin[0] + np.arange(nx) * pitch
    ys = origin[1] + np.arange(ny) * pitch
    zs = origin[2] + np.arange(nz) * pitch
    R = np.hypot(xs[:, None], zs[None, :])                    # (nx, nz)
    env[:, ys > y_top + 1e-9, :] = False                      # flat top cut
    if boss is not None:
        ysel = (ys >= float(boss[1]) - 1e-9) & (ys <= y_top + 1e-9)
        env[:, ysel, :] |= (R <= float(boss[0]))[:, None, :]
    shell = env & ~ndimage.binary_erosion(env, border_value=0)
    shell[:, ys < -1e-9, :] = False                           # only above the base
    in_hole_y = ys >= float(hole_y0) - 1e-9
    shell[:, in_hole_y, :] &= ~(R <= float(hole_radius) + 1.5 * pitch)[:, None, :]
    sx, sy, sz = np.nonzero(shell)
    if sx.size == 0:
        return None
    t_cav = sdf[sx, sy, sz].astype(float) - cav + 0.5 * pitch
    # The pour hole limits only side-facing outer surface (a top-face voxel beside the
    # hole opening is a lip, not a thin wall).
    hole_side = in_hole_y[sy] & (ys[sy] < float(y_top) - 1.5 * pitch)
    t_hole = np.where(hole_side, R[sx, sz] - float(hole_radius) + 0.5 * pitch, np.inf)
    t = np.minimum(t_cav, t_hole)
    i = int(np.argmin(t))
    if t_hole[i] < t_cav[i]:
        where = "around the pour hole"
    elif ys[sy[i]] >= float(y_top) - 2.0 * pitch:
        where = "the top face"
    else:
        where = "the side wall"
    return {"min_mm": float(max(0.0, t[i])), "y_src": float(ys[sy[i]]),
            "where": where, "pitch": pitch}


def wedge_prism_arrays(a0, a1, y0, y1, radius):
    """Vertices/faces of a closed pie-slice prism (apex on the Y axis)."""
    n = max(4, int(math.ceil(abs(a1 - a0) / (math.pi / 72.0))))
    ang = a0 + (a1 - a0) * np.arange(n + 1) / n
    arc = np.stack([np.sin(ang) * radius, np.cos(ang) * radius], axis=-1)
    verts = [[0.0, y0, 0.0]] + [[x, y0, z] for x, z in arc]
    verts += [[0.0, y1, 0.0]] + [[x, y1, z] for x, z in arc]
    verts = np.asarray(verts, dtype=float)
    b0, t0 = 0, n + 2
    bk = lambda k: 1 + k            # bottom arc point k
    tk = lambda k: n + 3 + k        # top arc point k
    faces = []
    for k in range(n):
        faces.append([b0, bk(k + 1), bk(k)])                     # bottom fan
        faces.append([t0, tk(k), tk(k + 1)])                     # top fan
        faces.append([bk(k), bk(k + 1), tk(k + 1)])              # curved side
        faces.append([bk(k), tk(k + 1), tk(k)])
    faces.append([b0, bk(0), tk(0)]); faces.append([b0, tk(0), t0])      # side at a0
    faces.append([b0, t0, tk(n)]);    faces.append([b0, tk(n), bk(n)])   # side at a1
    return verts, np.asarray(faces, dtype=np.int64)


def wedge_prism(a0, a1, y0, y1, radius):
    if abs(a1 - a0) >= 2 * math.pi - 1e-6:       # single full-circle "panel"
        cyl = vertical_cylinder(radius, y1 - y0, sections=144)
        cyl.apply_translation([0, (y0 + y1) / 2.0, 0])
        return cyl
    verts, faces = wedge_prism_arrays(a0, a1, y0, y1, radius)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    out.fix_normals()
    return out


def translated_wedge_prism(a0, a1, off0, off1, y0, y1, radius):
    """Extrude a sector bounded by two vertically translated cut planes.

    The seam planes remain vertical and retain their original directions, but
    each can move in the horizontal X/Z plane. The resulting convex footprint
    is clipped from a circular envelope by the two actual offset seam lines.
    """
    n = max(96, int(math.ceil(2 * math.pi * radius / 2.0)))
    poly = [(radius * math.cos(2 * math.pi * k / n),
             radius * math.sin(2 * math.pi * k / n)) for k in range(n)]

    def clip(points, a, offset, keep_le=True):
        # cross(direction(a), point) + offset = 0 is the translated seam.
        dx, dz = math.sin(a), math.cos(a)
        def value(pt):
            v = dx * pt[1] - dz * pt[0] + offset
            return v if keep_le else -v
        if not points:
            return []
        out=[]
        prev=points[-1]; fp=value(prev)
        for cur in points:
            fc=value(cur)
            if (fp <= 1e-9) != (fc <= 1e-9):
                t=fp/(fp-fc)
                out.append((prev[0]+t*(cur[0]-prev[0]), prev[1]+t*(cur[1]-prev[1])))
            if fc <= 1e-9:
                out.append(cur)
            prev,fp=cur,fc
        return out

    # Panel i lies counter-clockwise between its lower and upper seam.
    poly = clip(poly, a0, float(off0), True)
    poly = clip(poly, a1, float(off1), False)
    # Clipping can emit the same vertex twice when an original circle vertex
    # lies exactly on a seam. Remove consecutive duplicates before meshing.
    cleaned=[]
    for point in poly:
        if not cleaned or math.hypot(point[0]-cleaned[-1][0], point[1]-cleaned[-1][1]) > 1e-7:
            cleaned.append(point)
    if len(cleaned)>2 and math.hypot(cleaned[0][0]-cleaned[-1][0], cleaned[0][1]-cleaned[-1][1]) <= 1e-7:
        cleaned.pop()
    poly=cleaned
    if len(poly) < 3:
        raise ValueError("The moved cut planes leave a panel with no usable area. Move the planes closer to their original positions.")
    m=len(poly)
    verts=np.asarray([[x,y0,z] for x,z in poly]+[[x,y1,z] for x,z in poly],dtype=float)
    faces=[]
    # Convex polygon caps, triangulated as fans.
    for k in range(1,m-1):
        faces.append([0,k+1,k])
        faces.append([m,m+k,m+k+1])
    for k in range(m):
        j=(k+1)%m
        faces.append([k,j,m+j]); faces.append([k,m+j,m+k])
    out=trimesh.Trimesh(vertices=verts,faces=np.asarray(faces,dtype=np.int64),process=True)
    out.fix_normals()
    return out


def envelope_profile(fields, wall, clearance, height, boss=None,
                     angle_count=144, rows=96):
    """Radial profile (same format as radial_profile) of the uniform offset wall.

    `fields` come from offset_model(..., return_fields=True) run at
    clearance + wall.  The outer radius at every (height, angle) is the outermost
    occupied voxel along that ray, so it follows the real offset surface rather
    than the old smoothed vertex envelope.  It is stored the way the flange, nub
    and seal code expect it (outer + clearance + wall + reserve == wall surface).
    The base disc radius is the wall surface at the model's bottom minus `wall`,
    so the ring around the base is exactly `wall` thick.
    """
    mask = np.asarray(fields["offset_volume"], dtype=bool)
    pitch = float(fields["pitch"])
    origin = np.asarray(fields["origin"], dtype=float)
    nx, ny, nz = mask.shape
    ys = np.linspace(0.0, float(height), rows + 1)
    jj = np.rint((ys - origin[1]) / pitch).astype(int)
    valid_y = (jj >= 0) & (jj < ny)
    jj = np.clip(jj, 0, ny - 1)
    xmax = max(abs(origin[0]), abs(origin[0] + (nx - 1) * pitch))
    zmax = max(abs(origin[2]), abs(origin[2] + (nz - 1) * pitch))
    K = int(math.ceil(math.hypot(xmax, zmax) / pitch)) + 1
    r = np.arange(K + 1) * pitch
    angs = np.arange(angle_count) / angle_count * 2 * np.pi
    ii = np.rint((r[None, :] * np.sin(angs)[:, None] - origin[0]) / pitch).astype(int)
    ll = np.rint((r[None, :] * np.cos(angs)[:, None] - origin[2]) / pitch).astype(int)
    inb = (ii >= 0) & (ii < nx) & (ll >= 0) & (ll < nz)
    ii = np.clip(ii, 0, nx - 1); ll = np.clip(ll, 0, nz - 1)

    outer = np.full((rows + 1, angle_count), np.nan)
    for row in range(rows + 1):
        if not valid_y[row]:
            continue
        inside = mask[ii, jj[row], ll] & inb               # (angles, K+1)
        any_in = inside.any(axis=1)
        last = K - np.argmax(inside[:, ::-1], axis=1)      # outermost occupied sample
        outer[row, any_in] = r[last][any_in] + 0.5 * pitch
    if not np.isfinite(outer).any():
        raise RuntimeError("Could not measure the outer wall from the offset solid.")
    rr = np.arange(rows + 1)
    for a in range(angle_count):
        col = outer[:, a]; ok = np.isfinite(col)
        if ok.any():
            outer[:, a] = np.interp(rr, rr[ok], col[ok])   # clamps to nearest valid row
    outer[~np.isfinite(outer)] = np.nanmax(outer)
    # One conservative smoothing pass (never moves the wall inward).
    avg = (outer * 4 + np.roll(outer, 1, axis=1) + np.roll(outer, -1, axis=1)
           + np.vstack([outer[:1], outer[:-1]]) + np.vstack([outer[1:], outer[-1:]])) / 8
    outer = np.maximum(outer, avg)
    if boss is not None:                      # pour boss: (radius, source-y where it starts)
        outer[ys >= float(boss[1]) - 1e-6] = np.maximum(
            outer[ys >= float(boss[1]) - 1e-6], float(boss[0]))
    base_edge = np.maximum(2.0, outer[0] - float(wall))
    inner = np.tile(base_edge - float(clearance), (rows + 1, 1))
    reserve = max(5.0, float(wall) * 0.55)
    return {"inner": inner, "outer": outer - float(clearance) - float(wall) - reserve,
            "height": float(height), "angles": angle_count, "rows": rows}


def transform_to_export(mesh):
    """Application Y-up to printer STL Z-up, keeping the bottom at Z=0."""
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
    out.apply_translation([0, 0, -out.bounds[0,2]])
    return out


def transform_set_to_export(meshes):
    """Y-up to Z-up for several meshes at once, keeping their relative positions.

    transform_to_export() drops every mesh onto Z=0 individually, which is right
    for printing a part but undoes an exploded layout.  Here the whole set shares
    one offset so the lowest point of the set sits at Z=0.
    """
    outs = []
    for mesh in meshes:
        out = mesh.copy()
        out.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
        outs.append(out)
    low = min(float(o.bounds[0, 2]) for o in outs)
    for o in outs:
        o.apply_translation([0, 0, -low])
    return outs


def transform_to_preview(mesh):
    """Rotate a preview copy to Z-up while preserving its world position.

    Unlike STL export, the visualiser must not rebase each individual mesh to
    Z=0.  Doing so discards the source model's base_top/lift_y placement and
    makes the source appear to sink into or intersect the mould base.
    """
    out = mesh.copy()
    out.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    )
    return out


def mesh_footprint(mesh):
    """Union of every triangle projected onto the horizontal X/Z plane (shapely)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    tri = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces)][:, :, [0, 2]]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area2 = np.abs((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0]))
    tri = tri[area2 > 2e-7]
    if len(tri) == 0:
        raise ValueError("Could not extract a usable 2D outline from this STL.")
    try:
        import shapely
        polys = shapely.polygons(tri)           # shapely >= 2: vectorised
        footprint = shapely.union_all(polys)
    except (ImportError, AttributeError):
        footprint = unary_union([Polygon(t) for t in tri])
    return footprint.buffer(0)


def join_islands(shape, margin):
    """Connect separate polygons with bridges so one tray can hold them all.

    Repeatedly links the nearest remaining island to the growing cluster with a
    strip `2 * margin` wide. Returns (single_polygon, number_of_bridges).
    """
    from shapely.geometry import LineString
    from shapely.ops import nearest_points, unary_union

    geoms = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    if len(geoms) <= 1:
        return shape, 0
    cluster = geoms.pop(0)
    bridges = 0
    while geoms:
        k = min(range(len(geoms)), key=lambda i: cluster.distance(geoms[i]))
        g = geoms.pop(k)
        p, q = nearest_points(cluster, g)
        strip = LineString([(p.x, p.y), (q.x, q.y)]).buffer(max(1.0, margin))
        cluster = unary_union([cluster, g, strip]).buffer(0)
        bridges += 1
    if cluster.geom_type == "MultiPolygon":
        cluster = max(cluster.geoms, key=lambda g: g.area)
    return cluster, bridges


@dataclass
class MoldPart:
    name: str
    mesh: trimesh.Trimesh
    direction: tuple = (0.0, 0.0, 0.0)


class MoldGenerator:
    def __init__(self):
        self.source = None
        self.plug = None
        self.parts: list[MoldPart] = []
        self.profile = None
        self.lift_y = 0.0
        self.base_top = 0.0
        self.plug_base_clearance = 0.0
        self.cavity_volume_mm3 = 0.0
        self.pour_channel_volume_mm3 = 0.0
        self.total_resin_volume_mm3 = 0.0
        self.flat_stats = None  # filled by generate_flat_2d()
        self.min_wall_mm = None       # thinnest outer wall estimate (offset wall only)
        self.min_wall_where = None
        self.warnings: list[str] = []

    def generate(self, panels=4, top_extra=8, wall=8, base=6, clearance=3,
                 fill_diameter=12, fill_depth_percent=50, core=False, rim_width=2.5,
                 rim_thickness=1.5, wall_style="offset", key_width=3.5, key_depth=2.0,
                 progress=None, cut_angles=None, cut_positions=None):
        if self.source is None:
            raise ValueError("Load an STL first.")
        # Work on a private copy: generation recentres and transforms the mesh,
        # and the cut-plane range validation below needs its original extents.
        src = self.source.copy()
        panels = int(panels)
        if panels < 2 or panels > 8:
            raise ValueError("Panel count must be between 2 and 8.")
        if cut_angles is None:
            seam_angles = [i / panels * 2.0 * math.pi for i in range(panels)]
        else:
            try:
                raw_angles = [float(a) % 360.0 for a in cut_angles]
            except (TypeError, ValueError):
                raise ValueError("Cut plane angles must be valid numbers.")
            if len(raw_angles) != panels or not np.all(np.isfinite(raw_angles)):
                raise ValueError("The number of cut plane angles must match the panel count.")
            raw_angles.sort()
            gaps = [((raw_angles[(i + 1) % panels] - raw_angles[i]) % 360.0)
                    for i in range(panels)]
            if min(gaps) < 1.0:
                raise ValueError("Cut planes must remain at least 1 degree apart.")
            seam_angles = [math.radians(a) for a in raw_angles]
        # A complete vertical plane makes two opposing cuts. For even panel
        # counts, accept one position per intersecting plane and mirror that
        # position onto its opposite seam. The 5-panel legacy mode retains its
        # five independent seam positions because those cuts are not diametral.
        plane_count = panels // 2 if panels % 2 == 0 else panels
        if cut_positions is None:
            plane_positions = [(0.0, 0.0)] * plane_count
        else:
            try:
                plane_positions = [(float(v[0]), float(v[1])) for v in cut_positions]
            except (TypeError, ValueError, IndexError):
                raise ValueError("Each cut plane needs valid X and Y movement values.")
            if len(plane_positions) != plane_count or not np.all(np.isfinite(np.asarray(plane_positions))):
                raise ValueError(f"This {panels}-panel mould requires {plane_count} cut plane position(s).")
            if any(math.hypot(x, z) > 0.45 * max(src.extents[0], src.extents[2], 1.0) for x, z in plane_positions):
                raise ValueError("A cut plane has been moved too far from the model centre.")
        seam_offsets = ([plane_positions[i % plane_count] for i in range(panels)]
                        if panels % 2 == 0 else list(plane_positions))
        # The wall envelope, pour hole and panels are all built around the
        # vertical axis through the origin, so centre the (possibly multi-model)
        # source on its bounding box and sit it on Y=0.
        bb = src.bounds
        src.apply_translation([-(bb[0,0]+bb[1,0])/2.0, -bb[0,1], -(bb[0,2]+bb[1,2])/2.0])
        size = src.extents
        if np.any(size <= 0):
            raise ValueError("The STL has invalid or zero dimensions.")
        if progress: progress("Analysing the model…", 0.02)
        wall=max(3,float(wall)); base=max(3,float(base))
        # Keep the base underside at the local build-plate reference and make
        # its actual thickness exactly equal to the user's base setting.
        # After the mould parts are lifted onto the grid, base_top therefore
        # lands at Z=base (not at an arbitrary offset such as base - 0.5).
        y_min=1-base
        base_bottom=y_min
        actual_base=base
        base_top=base_bottom+actual_base
        self.base_top = base_top
        y_max=base_top+size[1]+max(0,float(top_extra))
        use_offset = str(wall_style) != "radial"
        profile = None if use_offset else radial_profile(src, size[1])
        self.profile = profile
        self.warnings = []
        if progress: progress("Building the cavity offset (the slow step on big models)…", 0.06)
        cutter, core_fields = offset_model(src, clearance, return_fields=True)
        if core_fields["pitch"] > 0.6:
            self.warnings.append(
                f"Large model: the cavity was built at {core_fields['pitch']:.2f} mm resolution, "
                "so fine surface detail in the silicone gap will be softened.")
        if not core:
            core_fields = None
        # The offset cutter encloses the expanded model envelope. Resin fills
        # only the gap between that envelope and the original STL, so subtract
        # the original solid volume from the offset volume.
        expanded_volume = abs(float(cutter.volume))
        source_volume = abs(float(src.volume))
        self.cavity_volume_mm3 = expanded_volume - source_volume
        if (not np.isfinite(self.cavity_volume_mm3) or
                self.cavity_volume_mm3 <= 0):
            raise ValueError(
                "Could not calculate a valid silicone gap volume. "
                "Check that the source STL is a closed, watertight solid "
                "and that the cavity clearance is greater than zero."
            )
        cutter.apply_translation([0,base_top,0])

        # Optional removable core for open cavities (cups, bowls, vases).
        core_mesh = core_cut = None
        if core:
            if progress: progress("Building removable core…", 0.22)
            core_mesh, core_cut, core_warnings = build_core(
                core_fields, top_y=float(size[1]), fill_diameter=fill_diameter)
            self.warnings.extend(core_warnings)
            core_fields = None
            if core_mesh is not None:
                core_mesh.apply_translation([0, base_top, 0])
                core_cut.apply_translation([0, base_top, 0])

        # Uniform-thickness outer wall: the cavity offset pushed out by `wall`, so the
        # mould wall follows the model instead of a radial envelope.
        env_full = None
        env_radius = 0.0
        # Locating keys: a vertical groove on the cavity surface along every seam, split
        # between the two panels, so the cast silicone gets matching ribs that seat in the
        # printed shell.  key_width is the finished slot width across the seam.
        key_width = max(0.0, float(key_width))
        key_depth = min(max(0.0, float(key_depth)), max(0.0, wall - 3.0))
        key_on = key_width > 0.05 and key_depth > 0.05
        key_deep = None
        if use_offset:
            if progress: progress("Building the uniform-thickness outer wall…", 0.24)
            hole_diameter = float(fill_diameter)
            if not np.isfinite(hole_diameter) or hole_diameter <= 0:
                raise ValueError("Fill hole diameter must be a positive number.")
            env, env_fields = offset_model(src, clearance + wall, return_fields=True, keep_sdf=True)
            # The wall cannot rise above the offset solid's own top.
            y_max = min(y_max, base_top + float(env.bounds[1, 1]))
            pour_depth = (max(1.0, min(100.0, float(fill_depth_percent))) / 100.0) * (y_max - y_min)
            boss_h = min(pour_depth, max(10.0, float(top_extra) + wall))
            boss_r = hole_diameter / 2.0 + max(3.0, wall * 0.5)
            profile = envelope_profile(env_fields, wall, clearance, y_max - base_top,
                                       boss=(boss_r, (y_max - boss_h) - base_top))
            self.profile = profile
            if key_on:
                try:
                    key_deep = surface_from_sdf(env_fields, clearance + 0.15 + key_depth)
                except Exception:
                    key_deep = None      # fall back to a separate offset below
            try:
                thin = wall_thickness_estimate(
                    env_fields, clearance, wall, y_max - base_top, hole_diameter / 2.0,
                    (y_max - pour_depth) - base_top, boss=(boss_r, (y_max - boss_h) - base_top))
            except Exception:
                thin = None            # the estimate is advisory; never fail the build over it
            if thin is not None:
                self.min_wall_mm = round(thin["min_mm"], 1)
                self.min_wall_where = thin["where"]
                if thin["min_mm"] < 2.0:
                    self.warnings.append(
                        f"Thin wall: about {thin['min_mm']:.1f} mm at {thin['where']} "
                        "(estimate). Walls under ~2 mm may print poorly or break; "
                        + ("raise Extra height above model or Wall thickness."
                           if thin["where"] != "around the pour hole"
                           else "use a smaller fill hole or a thicker wall."))
            env_fields = None
            env.apply_translation([0, base_top, 0])
            # Pour boss: keeps a solid collar around the sprue where the wall is thin.
            boss = vertical_cylinder(radius=boss_r, height=boss_h + 1.0, sections=96)
            boss.apply_translation([0, y_max - boss_h / 2.0 + 0.5, 0])
            env_full = boolean([env, boss], "union")
            env_radius = float(np.hypot(env_full.vertices[:, 0], env_full.vertices[:, 2]).max()) + 5.0

        # Keep the locating-slot extension aligned with the same pour-hole
        # diameter/depth settings in both offset and radial wall modes.
        hole_diameter = float(fill_diameter)
        if not np.isfinite(hole_diameter) or hole_diameter <= 0:
            raise ValueError("Fill hole diameter must be a positive number.")

        # Keep the locating-slot extension aligned with the same pour-hole
        # diameter/depth settings in both offset and radial wall modes.
        hole_diameter = float(fill_diameter)
        if not np.isfinite(hole_diameter) or hole_diameter <= 0:
            raise ValueError("Fill hole diameter must be a positive number.")

        key_cuts = [None] * max(1, int(panels))
        self.key_volume_mm3 = 0.0
        if key_on:
            try:
                if progress: progress("Cutting the locating keys…", 0.26)
                if key_deep is None:
                    key_deep = offset_model(src, clearance + key_depth)
                key_deep.apply_translation([0, base_top, 0])
                slab_r = float(np.hypot(key_deep.vertices[:, 0], key_deep.vertices[:, 2]).max()) + 5.0
                for k in range(int(panels)):
                    seam = seam_angles[k]
                    off_x, off_z = seam_offsets[k]
                    seam_offset = math.cos(seam) * off_x - math.sin(seam) * off_z
                    slab = key_slab(seam, base_top, y_max, slab_r, key_width, seam_offset)
                    kc = boolean([key_deep, slab], "intersection")
                    if kc is None or len(kc.faces) == 0:
                        continue
                    # Continue each cavity locating groove as a continuous slot
                    # to the mould top AND down into the pour-hole opening.  The
                    # previous extension started at the groove's highest point;
                    # when the pour hole ended lower down, that left a solid bridge
                    # at the bottom of the hole and made the groove look clipped.
                    kc_top = float(kc.bounds[1, 1])
                    hole_depth_for_keys = (
                        max(1.0, min(100.0, float(fill_depth_percent))) / 100.0
                    ) * (y_max - y_min)
                    hole_bottom_y = y_max - hole_depth_for_keys
                    rad = np.array([math.sin(seam), 0.0, math.cos(seam)])
                    near_top = kc.vertices[kc.vertices[:, 1] >= kc_top - 1.0]
                    r_ext = max(
                        float((near_top @ rad).max()) + 0.5,
                        hole_diameter / 2.0 + key_width + 1.0,
                        key_width,
                    )
                    # Start below the pour-hole bottom so the slot positively
                    # intersects the hole, rather than merely touching its edge.
                    ext_y0 = max(
                        base_top,
                        min(kc_top - 1.0, hole_bottom_y - 1.0),
                    )
                    ext = key_slab(seam, ext_y0, y_max + 0.5, r_ext, key_width, seam_offset)
                    kc = boolean([kc, ext], "union")
                    key_cuts[k] = kc
                    extra = boolean([kc, cutter], "difference")
                    if extra is not None and len(extra.faces):
                        self.key_volume_mm3 += abs(float(extra.volume))
                self.cavity_volume_mm3 += self.key_volume_mm3
            except Exception as exc:
                key_cuts = [None] * max(1, int(panels))
                self.key_volume_mm3 = 0.0
                self.warnings.append(f"Locating keys were skipped: {exc}")
            key_deep = None

        # Calculate the pour channel's additional volume. The channel is
        # counted only where it lies outside the expanded model envelope,
        # preventing overlap with the already-counted resin gap.
        hole_diameter = float(fill_diameter)
        if not np.isfinite(hole_diameter) or hole_diameter <= 0:
            raise ValueError("Fill hole diameter must be a positive number.")
        hole_depth = (max(1.0, min(100.0, float(fill_depth_percent))) / 100.0) * (y_max - y_min)
        pour_channel = vertical_cylinder(
            radius=hole_diameter / 2,
            height=hole_depth,
            sections=96,
        )
        # Measure only the portion inside the mold, not the extra 1 mm
        # cutter overrun used by the panel Boolean operation.
        pour_channel.apply_translation(
            [0, y_max - hole_depth / 2, 0]
        )
        channel_overlap = boolean(
            [pour_channel, cutter], "intersection"
        )
        channel_overlap_volume = (
            abs(float(channel_overlap.volume))
            if channel_overlap is not None and len(channel_overlap.faces)
            else 0.0
        )
        self.pour_channel_volume_mm3 = max(
            0.0, abs(float(pour_channel.volume)) - channel_overlap_volume
        )
        self.total_resin_volume_mm3 = (
            self.cavity_volume_mm3 + self.pour_channel_volume_mm3
        )

        if progress: progress("Building fitted base…", 0.28)
        base_blank=radial_prism(profile,base_bottom,base_top,clearance)
        parts=[]
        # External base rim (fused into the bottom of every wall panel).
        rim_width=max(0.0,float(rim_width))
        rim_on=rim_width>0.01
        rim_height=min(max(0.5,float(rim_thickness)),max(0.5,base-1.0)) if rim_on else 0.0
        # Base = fitted disc + a thin external rim fused around its edge at the
        # middle of the base thickness.  The same rim is Boolean-cut out of the
        # bottom of every wall panel as a matching groove (BASE_RIM_GAP clearance).
        rim_cut=None
        if rim_on:
            mid_y=base_bottom+actual_base/2
            rim_ring=base_tab_ring(profile,mid_y,rim_height,rim_width,clearance)
            rim_cut=base_tab_ring(profile,mid_y,rim_height,rim_width,clearance,
                                  root=1.5,grow=BASE_RIM_GAP)
            base_blank=boolean([base_blank,rim_ring],"union")
        base_mesh=base_blank

        for i in range(int(panels)):
            if progress: progress(f"Generating cavity panel {i+1}/{panels}…",
                                  0.30 + 0.65 * i / max(1, int(panels)))
            a0=seam_angles[i]; a1=seam_angles[(i+1)%panels]
            if a1 <= a0:
                a1 += 2.0 * math.pi
            # Plane movement is X/Z in the engine's horizontal plane. Only the
            # component normal to each plane changes its cut location.
            p0x,p0z=seam_offsets[i]; p1x,p1z=seam_offsets[(i+1)%panels]
            off0=math.cos(a0)*p0x-math.sin(a0)*p0z
            off1=math.cos(a1)*p1x-math.sin(a1)*p1z
            if use_offset:
                # Offset wall above the base, plus the fitted ring around the base disc.
                wedge = translated_wedge_prism(a0, a1, off0, off1, base_top, y_max, env_radius)
                upper = boolean([env_full, wedge], "intersection")
                ring = annular_panel(profile, a0, a1, y_min, base_top + 0.5,
                                     wall, clearance, base_top)
                ring_cut = translated_wedge_prism(a0, a1, off0, off1,
                                                  y_min - 0.5, base_top + 1.0, env_radius)
                ring = boolean([ring, ring_cut], "intersection")
                blank = boolean([ring, upper], "union")
            else:
                blank=annular_panel(profile,a0,a1,y_min,y_max,wall,clearance,base_top)
                moved_wedge = translated_wedge_prism(a0, a1, off0, off1,
                                                     y_min - 0.5, y_max + 0.5, env_radius)
                blank = boolean([blank, moved_wedge], "intersection")
            panel=boolean([blank,cutter],"difference")
            for seam in (i, (i + 1) % int(panels)):
                if key_cuts[seam] is not None:
                    panel = boolean([panel, key_cuts[seam]], "difference")
            if core_cut is not None:
                # Bore the panel so the removable core passes through it.
                panel=boolean([panel,core_cut],"difference")
            # Flange-mounted alignment nubs below provide panel registration.
            # Contoured external clip flanges. Their full height follows the
            # smoothed outer wall profile instead of using a rectangular block.
            # Opposing tabs are shifted tangentially by half their thickness,
            # matching the HTML generator's edge-to-edge arrangement.
            flange_projection = 12.0
            flange_thickness = 3.0
            flange_offset = flange_thickness / 2
            # Each seam uses a male/female pair across the touching side
            # faces of the offset flanges.  The two locating nubs sit near the
            # top and bottom of the flange and are centred across its radial
            # width (the middle of the flange's projection).
            flange_nub_radius = 1.5
            flange_hole_radius = 1.8
            flange_nub_depth = 2.8
            flange_end_inset = min(
                (y_max - y_min) * 0.2,
                max(4.0, flange_nub_radius + 1.5)
            )
            flange_nub_heights = (
                y_min + flange_end_inset,
                y_max - flange_end_inset,
            )

            for edge, angle in ((0, a0), (1, a1)):
                tangential_offset = flange_offset if edge == 0 else -flange_offset
                flange = contoured_clip_flange(
                    profile, angle, y_min, y_max, base_top,
                    clearance=clearance, wall=wall,
                    projection=flange_projection,
                    thickness=flange_thickness,
                    tangential_offset=tangential_offset,
                )

                # Add a continuous perimeter seal to each flange mating face.
                # Edge 1 carries the raised bead; edge 0 receives its matching
                # slightly oversized groove, just like the base rim joint.
                if rim_on:
                    seal = flange_seal_ring(
                        angle, profile, y_min, y_max, base_top,
                        clearance=clearance, wall=wall,
                        projection=flange_projection,
                        tangential_face=0.0, rim_width=rim_width,
                        rim_thickness=rim_thickness,
                        grow=0.12 if edge == 0 else 0.0,
                        line_y_min=flange_nub_heights[0],
                        line_y_max=flange_nub_heights[1],
                    )
                    if seal is not None:
                        if edge == 1:
                            flange = boolean([flange, seal], "union")
                        else:
                            flange = boolean([flange, seal], "difference")

                # At every seam, the right-hand flange (edge 1) carries
                # outward-facing male nubs.  The next panel's left-hand
                # flange (edge 0) receives matching blind female sockets.
                tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)])
                radial = np.array([math.sin(angle), 0.0, math.cos(angle)])
                side_tangent = (
                    tangential_offset + flange_thickness / 2
                    if edge == 1 else
                    tangential_offset - flange_thickness / 2
                )

                for nub_y in flange_nub_heights:
                    source_y = float(np.clip(nub_y - base_top, 0, profile["height"]))
                    panel_outer = (
                        profile_radius(profile, angle, source_y, True)
                        + max(0.0, clearance) + wall
                        + max(5.0, wall * 0.55)
                    )
                    nub_radial = panel_outer + flange_projection / 2
                    face_point = (
                        radial * nub_radial
                        + np.array([0.0, nub_y, 0.0])
                        + tangent * side_tangent
                    )

                    if edge == 1:
                        # Start slightly inside the flange face to guarantee
                        # a sound union, then project into the mating flange.
                        male_start = face_point - tangent * 0.35
                        male = cylinder_between(
                            flange_nub_radius, flange_nub_depth,
                            male_start, tangent
                        )
                        flange = boolean([flange, male], "union")
                    else:
                        # Matching socket, with a small radial clearance.
                        socket_start = face_point - tangent * 0.25
                        socket = cylinder_between(
                            flange_hole_radius, flange_nub_depth + 0.5,
                            socket_start, tangent
                        )
                        flange = boolean([flange, socket], "difference")

                # Carry the complete flange, seal and locating features with
                # the translated mating cut plane.
                seam_offset = off0 if edge == 0 else off1
                flange.apply_translation([math.cos(angle) * seam_offset,
                                          0.0, -math.sin(angle) * seam_offset])
                panel = boolean([panel, flange], "union")

            if rim_cut is not None:
                # Boolean the base rim out of the wall bottoms: a groove the
                # base rim seats into.
                panel=boolean([panel,rim_cut],"difference")

            # Cut the adjustable silicone pour channel vertically through the
            # centre of the assembled mold, matching the HTML implementation.
            # trimesh cylinders are Z-axis by default, so use the Y-axis helper.
            hole_diameter = float(fill_diameter)
            if not np.isfinite(hole_diameter) or hole_diameter <= 0:
                raise ValueError("Fill hole diameter must be a positive number.")
            hole_depth = (max(1.0, min(100.0, float(fill_depth_percent))) / 100.0) * (y_max - y_min)
            hole = vertical_cylinder(
                radius=hole_diameter / 2,
                height=hole_depth + 1.0,
                sections=64,
            )
            hole.apply_translation([0, y_max - hole_depth / 2 + 0.5, 0])
            panel = boolean([panel, hole], "difference")
            parts.append(MoldPart(f"Mold_Panel_{i+1:02d}",panel,
                                  (math.sin((i+.5)/panels*2*np.pi),0,
                                   math.cos((i+.5)/panels*2*np.pi))))

        # Horizontal base-to-wall locating stubs/sockets removed.

        # The plug is the untouched source model (no pin holes).
        plug=src.copy()

        # Align all mould parts to the build plate.
        all_meshes=[base_mesh]+[p.mesh for p in parts]
        lowest=min(float(m.bounds[0,1]) for m in all_meshes)
        # Put the actual underside of the mould base exactly on the preview
        # build-plate grid (Z=0 after the Y-up -> Z-up preview rotation).
        # The previous -0.2 mm offset left the base straddling the grid plane,
        # making the source model look as though it was sitting on the grid.
        self.lift_y = -lowest
        for m in all_meshes:
            m.apply_translation([0, self.lift_y, 0])
        if core_mesh is not None:
            core_mesh.apply_translation([0, self.lift_y, 0])

        # Place the duck/plug at the top of the base.  The base underside is
        # on the build-plate grid, so its top is exactly `base` millimetres
        # above the grid.  Do not leave the plug at its imported Z=0 height.
        plug_clearance = 0.0
        target_plug_bottom = float(base_top + self.lift_y + plug_clearance)
        current_plug_bottom = float(plug.bounds[0, 1])
        plug.apply_translation([0, target_plug_bottom - current_plug_bottom, 0])

        # Retain a measurable placement record for the preview/API and for
        # validating that the plug is above the base rather than intersecting.
        self.plug = plug
        self.plug_base_clearance = float(plug.bounds[0, 1] - (base_top + self.lift_y))
        self.parts=[MoldPart("Mold_Base",base_mesh)]+parts
        if core_mesh is not None:
            # Core lifts straight up out of the mould when exploded.
            self.parts.append(MoldPart("Mold_Core",core_mesh,(0.0,1.0,0.0)))
        bad = [p.name for p in self.parts if not p.mesh.is_watertight]
        if bad:
            self.warnings.append("These parts are not watertight and may slice badly: "
                                 + ", ".join(bad) + ".")
        if progress: progress("Done.", 1.0)
        return self.parts

    def generate_flat_2d(self, wall=8.0, base=6.0, top_extra=8.0, margin=10.0,
                         label_tab=True, progress=None):
        """Create a contour-following flat pour mould around the STL footprint.

        The well, perimeter wall and base follow the projected X/Z silhouette
        of the source model instead of using a rectangular box.
        """
        if self.source is None:
            raise ValueError("Load an STL first.")
        from shapely.geometry import Polygon
        from shapely.ops import unary_union, triangulate

        src = self.source.copy()
        sx, sy, sz = map(float, src.extents)
        if min(sx, sy, sz) <= 0:
            raise ValueError("The STL has invalid or zero dimensions.")

        wall = max(2.0, float(wall))
        base = max(2.0, float(base))
        margin = max(2.0, float(margin))
        wall_height = sy + max(0.0, float(top_extra))

        if progress:
            progress("Tracing the model outline…", 0.1)

        # The union of every projected triangle describes the model's actual
        # footprint, including concave outlines, rather than its bounds.
        footprint = mesh_footprint(src)
        if footprint.is_empty:
            raise ValueError("The STL silhouette is empty.")
        # A single continuous pour tray is required. Islands closer than
        # 2 x margin merge on their own; anything further apart is bridged so
        # no model is left outside the tray.
        inner_shape = footprint.buffer(margin, join_style=1).buffer(0)
        inner_shape, bridges = join_islands(inner_shape, margin)
        if bridges:
            self.warnings.append(
                f"The models are far apart, so {bridges} bridge{'s were' if bridges > 1 else ' was'} "
                "added to join them into one tray. Move them closer together (Transform tab) "
                "for a smaller mould and less silicone.")
        # Use the exterior silhouette for a clean, manufacturable tray wall.
        inner_shape = Polygon(inner_shape.exterior).buffer(0)
        outer_shape = inner_shape.buffer(wall, join_style=1).buffer(0)
        if outer_shape.geom_type == "MultiPolygon":
            outer_shape = max(outer_shape.geoms, key=lambda g: g.area)
        outer_shape = Polygon(outer_shape.exterior).buffer(0)

        def polygon_faces(poly, y, reverse=False):
            """Triangulate a polygon cap, retaining only triangles inside it."""
            points = []
            faces = []
            for tri in triangulate(poly):
                if not poly.covers(tri.representative_point()):
                    continue
                coords = list(tri.exterior.coords)[:3]
                offset = len(points)
                points.extend([[x, y, z] for x, z in coords])
                faces.append([offset, offset + 1, offset + 2])
            if reverse:
                faces = [f[::-1] for f in faces]
            return points, faces

        def extruded_polygon(poly, y0, y1):
            """Watertight vertical extrusion of a polygon without holes."""
            poly = Polygon(poly.exterior)
            ring = list(poly.exterior.coords)[:-1]
            n = len(ring)
            verts = [[x, y0, z] for x, z in ring]
            verts += [[x, y1, z] for x, z in ring]
            faces = []
            p0, f0 = polygon_faces(poly, y0, reverse=True)
            p1, f1 = polygon_faces(poly, y1, reverse=False)
            # Cap triangulation uses independent vertices; append those caps.
            base0 = len(verts)
            verts.extend(p0)
            faces.extend([[base0 + a, base0 + b, base0 + c] for a,b,c in f0])
            base1 = len(verts)
            verts.extend(p1)
            faces.extend([[base1 + a, base1 + b, base1 + c] for a,b,c in f1])
            for i in range(n):
                j = (i + 1) % n
                faces.extend([[i, j, n + j], [i, n + j, n + i]])
            mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                                   faces=np.asarray(faces, dtype=np.int64),
                                   process=True)
            mesh.fix_normals()
            return mesh

        def wall_ring_mesh(inner, outer, y0, y1):
            """Build a closed extruded ring: top/bottom annuli and both sides."""
            ring_poly = outer.difference(inner).buffer(0)
            if ring_poly.geom_type == "MultiPolygon":
                ring_poly = unary_union(list(ring_poly.geoms))
            verts, faces = [], []
            # Triangulated annular caps.
            for y, reverse in ((y0, True), (y1, False)):
                for tri in triangulate(ring_poly):
                    if not ring_poly.covers(tri.representative_point()):
                        continue
                    coords = list(tri.exterior.coords)[:3]
                    base_idx = len(verts)
                    verts.extend([[x, y, z] for x, z in coords])
                    face = [base_idx, base_idx+1, base_idx+2]
                    faces.append(face[::-1] if reverse else face)
            # Vertical faces on every exterior and interior boundary.
            polygons = list(ring_poly.geoms) if ring_poly.geom_type == "MultiPolygon" else [ring_poly]
            for component in polygons:
                for boundary in [component.exterior, *component.interiors]:
                    coords = list(boundary.coords)
                    for a, b in zip(coords[:-1], coords[1:]):
                        idx = len(verts)
                        verts.extend([[a[0], y0, a[1]], [b[0], y0, b[1]],
                                      [b[0], y1, b[1]], [a[0], y1, a[1]]])
                        faces.extend([[idx, idx+1, idx+2], [idx, idx+2, idx+3]])
            mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                                   faces=np.asarray(faces, dtype=np.int64),
                                   process=True)
            mesh.fix_normals()
            return mesh

        if progress:
            progress("Forming base and perimeter wall…", 0.5)
        # Build the tray from the grid (Y=0) up to the exact user-specified
        # base thickness.  The duck plug starts on that top face, so it is
        # raised by the full base thickness instead of remaining on the grid.
        base_top = float(base)
        base_mesh = extruded_polygon(outer_shape, 0.0, base_top)
        walls_mesh = wall_ring_mesh(inner_shape, outer_shape, base_top,
                                    base_top + wall_height)
        plug = src.copy()
        plug.apply_translation([0.0, base_top, 0.0])
        # Label tab on the top edge carrying the silicone amount to fill the
        # well (same figure the app reports as "Total silicone required").
        label_meshes = []
        label_text = None
        if label_tab:
            silicone_ml = max(0.0, float(inner_shape.area) * wall_height
                              - abs(float(src.volume))) / 1000.0
            label_text = (f"{silicone_ml:.0f}" if silicone_ml >= 100
                          else f"{silicone_ml:.1f}") + " ML"
            label_meshes = build_label_tab(outer_shape, wall, base_top, label_text)
        if progress: progress("Fusing the tray and model…", 0.75)
        tray = boolean([base_mesh, walls_mesh, plug] + label_meshes, "union")
        tray.fix_normals()
        if not tray.is_watertight:
            raise RuntimeError("The contour-following 2D mould did not produce a watertight solid.")

        # Lift the complete flat mould clearly above the preview build-plate.
        # A 0.2 mm offset was too small for the visualiser's grid/depth buffer
        # and made the tray appear to intersect or sit below the grid.  Keep a
        # full 1.0 mm clearance so the underside is visibly above Z=0 after
        # the Y-up -> Z-up preview transform.  The plug receives the identical
        # translation so it remains aligned with the mould.
        grid_clearance = 1.0
        actual_lift = grid_clearance - float(tray.bounds[0, 1])
        tray.apply_translation([0.0, actual_lift, 0.0])
        plug.apply_translation([0.0, actual_lift, 0.0])

        # Silicone fills the contoured well around the raised STL plug.
        well_volume = float(inner_shape.area) * wall_height
        stl_volume = abs(float(src.volume))
        self.cavity_volume_mm3 = max(0.0, well_volume - stl_volume)
        self.pour_channel_volume_mm3 = 0.0
        self.total_resin_volume_mm3 = self.cavity_volume_mm3
        minx, minz, maxx, maxz = inner_shape.bounds
        self.flat_stats = {
            "footprint_area_mm2": float(inner_shape.area),
            "footprint_x_mm": float(maxx - minx),
            "footprint_z_mm": float(maxz - minz),
            "fill_depth_mm": float(wall_height),
            "well_volume_mm3": float(well_volume),
            "stl_volume_mm3": float(stl_volume),
            "label_text": label_text,
        }
        self.base_top = base_top + actual_lift
        self.lift_y = actual_lift
        self.profile = None
        self.plug = plug
        self.parts = [MoldPart("Flat_2D_Contoured_Negative_Mold", tray)]
        if progress: progress("Done.", 1.0)
        return self.parts
