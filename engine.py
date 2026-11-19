"""
MAD HAT — Negative Mold Generator (Python desktop edition)

Install:
    py -m pip install PySide6 pyvista pyvistaqt trimesh manifold3d numpy scipy shapely

Run:
    py MadHat_Negative_Mold_Generator.py

STL input is expected to be Z-up. Generated STL exports are Z-up.
Boolean operations use the Manifold3D engine via trimesh.
"""
from __future__ import annotations

import math
import os
import sys
import zipfile
import json
import io
import traceback
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


def radial_profile(mesh, height, angle_count=144, y_count=48):
    """Conservative radial envelope of the model at sampled heights/angles."""
    v = mesh.vertices
    radii = np.full((y_count + 1, angle_count), np.nan)
    yi = np.clip(np.rint(v[:, 1] / max(height, 1e-8) * y_count).astype(int), 0, y_count)
    angles = (np.arctan2(v[:, 0], v[:, 2]) + 2 * np.pi) % (2 * np.pi)
    ai = np.floor(angles / (2 * np.pi) * angle_count).astype(int) % angle_count
    rr = np.hypot(v[:, 0], v[:, 2])
    for row, col, radius in zip(yi, ai, rr):
        if not np.isfinite(radii[row, col]) or radius > radii[row, col]:
            radii[row, col] = radius

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


def offset_model(mesh, clearance, return_fields=False):
    """Build a robust outward offset using a filled voxel distance field.

    Vertex-normal offsets can fold over on dense, highly curved meshes.  A
    volumetric dilation avoids those self-intersections and gives the Boolean
    engine a single, watertight cutter.  The voxel pitch is capped at 0.5 mm
    for useful detail without creating an excessive memory footprint.
    """
    from scipy import ndimage

    distance = max(0.0, float(clearance)) + 0.15
    if not np.isfinite(distance):
        raise ValueError("Cavity clearance must be a finite number.")

    extents = np.asarray(mesh.extents, dtype=float)
    if np.any(~np.isfinite(extents)) or np.any(extents <= 0):
        raise ValueError("Cannot offset a mesh with invalid dimensions.")

    # Keep the grid fine enough to represent the clearance, while limiting
    # the total number of voxels for larger source models.
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

        # Add enough empty space around the source for the complete offset.
        pad = int(math.ceil(distance / pitch)) + 3
        padded = np.pad(occupied, pad, mode="constant", constant_values=False)

        # Distance from every empty voxel centre to the filled source volume.
        # Include half a voxel diagonal to compensate for centre sampling and
        # produce a conservative outward clearance.
        outside_distance = ndimage.distance_transform_edt(~padded, sampling=pitch)
        offset_volume = padded | (outside_distance <= distance + pitch * 0.5)

        out = trimesh.voxel.ops.matrix_to_marching_cubes(
            offset_volume, pitch=pitch
        )
        # matrix_to_marching_cubes returns coordinates relative to the padded
        # array. Restore the source voxel grid's world-space origin.
        grid_origin = np.asarray(surface.transform[:3, 3], dtype=float)
        out.apply_translation(grid_origin - pad * pitch)
        out.remove_unreferenced_vertices()
        out.process(validate=True)
        out.fix_normals()

        if (not out.is_watertight or not out.is_winding_consistent or
                not np.isfinite(out.vertices).all() or out.volume <= 0):
            raise RuntimeError(
                "The volumetric cavity offset did not produce a valid closed solid."
            )
        if return_fields:
            fields = {
                "padded": padded,                      # filled source voxels
                "offset_volume": offset_volume,        # source + clearance
                "pitch": float(pitch),
                "origin": grid_origin - pad * pitch,   # world pos of voxel [0,0,0]
            }
            return out, fields
        return out
    except Exception as exc:
        raise RuntimeError(
            f"Could not create a clean {distance:.2f} mm cavity offset: {exc}"
        ) from exc


def annular_panel(profile, a0, a1, y_min, y_max, wall, clearance, base_top):
    na, ny = 192, 128
    verts = []
    faces = []
    grid = (na + 1) * (ny + 1)
    reserve = max(5.0, wall * .55)
    for layer in range(2):
        for j in range(ny + 1):
            y = y_min + (y_max-y_min) * j/ny
            for i in range(na + 1):
                a = a0 + (a1-a0) * i/na
                source_y = y - base_top
                if y <= base_top + 1e-6:
                    ri = profile_radius(profile, a, 0) + clearance
                else:
                    ri = .25
                ro = profile_radius(profile, a, source_y, True) + clearance + wall + reserve
                r = ri if layer == 0 else ro
                verts.append([math.sin(a)*max(.2, r), y, math.cos(a)*max(.2, r)])

    def idx(layer, i, j):
        return layer*grid + j*(na+1)+i

    def quad(a,b,c,d):
        faces.extend([[a,b,d],[b,c,d]])

    for layer in range(2):
        for j in range(ny):
            for i in range(na):
                a,b,c,d = idx(layer,i,j),idx(layer,i+1,j),idx(layer,i+1,j+1),idx(layer,i,j+1)
                if layer == 0:
                    quad(a,d,c,b)
                else:
                    quad(a,b,c,d)
    for j in range(ny):
        quad(idx(0,0,j),idx(1,0,j),idx(1,0,j+1),idx(0,0,j+1))
        quad(idx(0,na,j),idx(0,na,j+1),idx(1,na,j+1),idx(1,na,j))
    for i in range(na):
        quad(idx(0,i,0),idx(1,i,0),idx(1,i+1,0),idx(0,i+1,0))
        quad(idx(0,i,ny),idx(1,i,ny),idx(1,i+1,ny),idx(0,i+1,ny))
    result = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)
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

    # 4. Pour hole through the core: on the axis if there is room, else at the
    #    widest part of the stem.
    edt = ndimage.distance_transform_edt(stem2d) * pitch
    wall = 2.0
    r_want = max(1.0, float(fill_diameter) / 2.0)
    i0 = int(round((0.0 - origin[0]) / pitch)); k0 = int(round((0.0 - origin[2]) / pitch))
    d_axis = edt[i0, k0] if (0 <= i0 < edt.shape[0] and 0 <= k0 < edt.shape[1]) else 0.0
    if d_axis - wall >= 1.5:
        hi, hk, d = i0, k0, d_axis
    else:
        hi, hk = np.unravel_index(int(np.argmax(edt)), edt.shape)
        d = edt[hi, hk]
    hole_r = min(r_want, d - wall)
    hole = None
    if hole_r >= 1.0:
        hole = (origin[0] + hi * pitch, origin[2] + hk * pitch, float(hole_r))
        if hole_r < r_want - 0.01:
            warnings.append(f"Core: pour hole reduced to {2*hole_r:.1f} mm diameter to fit the opening.")
    else:
        warnings.append("Core: the opening is too narrow for a pour hole through the core, "
                        "so none was cut. Silicone cannot be poured with this core.")

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

    if info["hole"] is not None:
        hx, hz, hr = info["hole"]
        y0 = float(core.bounds[0, 1]) - 1.0
        y1 = top_y + cap_height + 1.0
        hole = vertical_cylinder(radius=hr, height=y1 - y0, sections=48)
        hole.apply_translation([hx, (y0 + y1) / 2, hz])
        core = boolean([core, hole], "difference")
        if len(core.split(only_watertight=False)) > 1:
            warnings.append("Core: the pour hole splits the core into separate pieces. "
                            "Reduce the fill hole diameter.")
    core.fix_normals()
    return core, cut, warnings


def transform_to_export(mesh):
    """Application Y-up to printer STL Z-up, keeping the bottom at Z=0."""
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
    out.apply_translation([0, 0, -out.bounds[0,2]])
    return out


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
        self.warnings: list[str] = []

    def load_stl(self, path):
        mesh = trimesh.load(path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("The selected STL does not contain a triangle mesh.")
        # STL is Z-up; rotate into the generator's Y-up coordinates.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2, [1,0,0]))
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        mesh.apply_translation([-mesh.bounds.mean(axis=0)[0], -mesh.bounds[0,1],
                                -mesh.bounds.mean(axis=0)[2]])
        if not mesh.is_watertight:
            raise ValueError("The uploaded STL is not watertight. Repair it in a mesh tool and retry.")
        self.source = mesh
        self.parts.clear()
        return mesh.extents

    def generate(self, panels=4, top_extra=8, wall=8, base=6, clearance=3,
                 fill_diameter=12, fill_depth_percent=50, core=False, rim_width=2.5,
                 rim_thickness=1.5, progress=None):
        if self.source is None:
            raise ValueError("Load an STL first.")
        src = self.source.copy()
        size = src.extents
        if np.any(size <= 0):
            raise ValueError("The STL has invalid or zero dimensions.")
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
        profile=radial_profile(src,size[1])
        self.profile=profile
        self.warnings = []
        if core:
            cutter, core_fields = offset_model(src, clearance, return_fields=True)
        else:
            cutter = offset_model(src, clearance)
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
            if progress: progress("Building removable core…")
            core_mesh, core_cut, core_warnings = build_core(
                core_fields, top_y=float(y_max - base_top), fill_diameter=fill_diameter)
            self.warnings.extend(core_warnings)
            core_fields = None
            if core_mesh is not None:
                core_mesh.apply_translation([0, base_top, 0])
                core_cut.apply_translation([0, base_top, 0])

        # Calculate the pour channel's additional volume. The channel is
        # counted only where it lies outside the expanded model envelope,
        # preventing overlap with the already-counted resin gap.
        hole_diameter = max(2.0, min(50.0, float(fill_diameter)))
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

        if progress: progress("Building fitted base…")
        base_blank=radial_prism(profile,base_bottom,base_top,clearance)
        parts=[]
        # External base rim (fused into the bottom of every wall panel).
        rim_width=max(0.0,float(rim_width))
        rim_on=rim_width>0.01
        rim_height=min(max(0.5,float(rim_thickness)),max(0.5,base-1.0)) if rim_on else 0.0
        # Base = fitted disc + a thin external rim fused around its edge at the
        # middle of the base thickness.  The same rim is Boolean-cut out of the
        # bottom of every wall panel as a matching groove (0.2 mm clearance).
        rim_cut=None
        if rim_on:
            mid_y=base_bottom+actual_base/2
            rim_ring=base_tab_ring(profile,mid_y,rim_height,rim_width,clearance)
            rim_cut=base_tab_ring(profile,mid_y,rim_height,rim_width,clearance,
                                  root=1.5,grow=0.2)
            base_blank=boolean([base_blank,rim_ring],"union")
        base_mesh=base_blank

        for i in range(int(panels)):
            if progress: progress(f"Generating cavity panel {i+1}/{panels}…")
            a0=i/panels*2*np.pi; a1=(i+1)/panels*2*np.pi
            blank=annular_panel(profile,a0,a1,y_min,y_max,wall,clearance,base_top)
            panel=boolean([blank,cutter],"difference")
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

                panel = boolean([panel, flange], "union")

            if rim_cut is not None:
                # Boolean the base rim out of the wall bottoms: a groove the
                # base rim seats into.
                panel=boolean([panel,rim_cut],"difference")

            # Cut the adjustable silicone pour channel vertically through the
            # centre of the assembled mold, matching the HTML implementation.
            # trimesh cylinders are Z-axis by default, so use the Y-axis helper.
            hole_diameter = max(2.0, min(50.0, float(fill_diameter)))
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
        return self.parts

    def generate_flat_2d(self, wall=8.0, base=6.0, top_extra=8.0, margin=10.0,
                         progress=None):
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
            progress("Tracing STL silhouette and building contoured mould…")

        # Project every surface triangle onto the horizontal X/Z plane.
        # The union of these projected triangles describes the model's
        # actual footprint, including concave outlines, rather than its bounds.
        projected = []
        vertices = np.asarray(src.vertices)
        for face in np.asarray(src.faces):
            points = vertices[face][:, [0, 2]]
            poly = Polygon(points)
            if poly.is_valid and poly.area > 1e-7:
                projected.append(poly)
        if not projected:
            raise ValueError("Could not extract a usable 2D outline from this STL.")

        footprint = unary_union(projected).buffer(0)
        if footprint.is_empty:
            raise ValueError("The STL silhouette is empty.")
        # A single continuous pour tray is required. If the projection has
        # separate islands, join nearby islands using the requested margin.
        inner_shape = footprint.buffer(margin, join_style=1).buffer(0)
        if inner_shape.geom_type == "MultiPolygon":
            inner_shape = unary_union(list(inner_shape.geoms)).buffer(0)
        # Use the exterior silhouette for a clean, manufacturable tray wall.
        if inner_shape.geom_type == "MultiPolygon":
            inner_shape = max(inner_shape.geoms, key=lambda g: g.area)
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
            progress("Forming contoured base and perimeter wall…")
        # Build the tray from the grid (Y=0) up to the exact user-specified
        # base thickness.  The duck plug starts on that top face, so it is
        # raised by the full base thickness instead of remaining on the grid.
        base_top = float(base)
        base_mesh = extruded_polygon(outer_shape, 0.0, base_top)
        walls_mesh = wall_ring_mesh(inner_shape, outer_shape, base_top,
                                    base_top + wall_height)
        plug = src.copy()
        plug.apply_translation([0.0, base_top, 0.0])
        tray = boolean([base_mesh, walls_mesh, plug], "union")
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
        self.cavity_volume_mm3 = max(0.0, well_volume - abs(float(src.volume)))
        self.pour_channel_volume_mm3 = 0.0
        self.total_resin_volume_mm3 = self.cavity_volume_mm3
        self.base_top = base_top + actual_lift
        self.lift_y = actual_lift
        self.profile = None
        self.plug = plug
        self.parts = [MoldPart("Flat_2D_Contoured_Negative_Mold", tray)]
        return self.parts

    def export_part(self, index, path):
        transform_to_export(self.parts[index].mesh).export(path, file_type="stl")

    def export_zip(self, path, exploded=0):
        max_size=max(self.source.extents)
        dist=(exploded/100)*(8+max(12,max_size*.12))
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as zf:
            for i,part in enumerate(self.parts):
                mesh=part.mesh.copy()
                if i:
                    mesh.apply_translation(np.array(part.direction)*dist)
                out=transform_to_export(mesh)
                zf.writestr(part.name+".stl",out.export(file_type="stl"))
