"""Procedural geometry.

Vertices are interleaved: position(3) + normal(3) + color(3) + emissive(1)
+ material(2). ``emissive`` is 0 for normally-lit surfaces and up to 1 for
self-illuminated ones (e.g. destination signs, headlights, the odd lit
window). ``material`` packs (gloss, translucency): gloss 0 = rough matte
(soil, grass), 1 = polished (train paint, glass, rail steel); translucency
lets sunlight bleed through thin volumes (tree canopies).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

FLOATS_PER_VERTEX = 12


@dataclass
class Mesh:
    vertices: np.ndarray  # shape (N, 12) float32
    indices: np.ndarray   # shape (M,) uint32

    @property
    def index_count(self) -> int:
        return int(self.indices.shape[0])

    def vertex_bytes(self) -> bytes:
        return np.ascontiguousarray(self.vertices, dtype=np.float32).tobytes()

    def index_bytes(self) -> bytes:
        return np.ascontiguousarray(self.indices, dtype=np.uint32).tobytes()


# Six faces: (normal, four corner offsets in CCW order seen from outside)
_FACES = [
    ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),      # +Z
    ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),  # -Z
    ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]),       # +X
    ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),  # -X
    ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),       # +Y
    ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),  # -Y
]


def make_box(size=(1.0, 1.0, 1.0), color=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0),
             emissive=0.0, gloss=0.0, translucency=0.0) -> Mesh:
    """Axis-aligned box. ``size`` is the full extent along (x, y, z)."""
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = center
    verts = []
    idx = []
    for normal, corners in _FACES:
        base = len(verts)
        for ox, oy, oz in corners:
            verts.append(
                [cx + ox * hx, cy + oy * hy, cz + oz * hz, *normal, *color,
                 emissive, gloss, translucency]
            )
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_quad(corners, normal, color, emissive=0.0, gloss=0.0, translucency=0.0) -> Mesh:
    """A single quad from four CCW corner points (glm.vec3-like)."""
    verts = [
        [c[0], c[1], c[2], *normal, *color, emissive, gloss, translucency]
        for c in corners
    ]
    idx = [0, 1, 2, 0, 2, 3]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_ground(size, center=(0.0, 0.0), color=(0.3, 0.4, 0.3), y=0.0, emissive=0.0,
                gloss=0.0) -> Mesh:
    """A flat ground quad on the XZ plane, normal +Y."""
    cx, cz = center
    h = size / 2
    n = (0.0, 1.0, 0.0)
    verts = [
        [cx - h, y, cz - h, *n, *color, emissive, gloss, 0.0],
        [cx + h, y, cz - h, *n, *color, emissive, gloss, 0.0],
        [cx + h, y, cz + h, *n, *color, emissive, gloss, 0.0],
        [cx - h, y, cz + h, *n, *color, emissive, gloss, 0.0],
    ]
    idx = [0, 1, 2, 0, 2, 3]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_ellipsoid(radii=(1.0, 1.0, 1.0), color=(1.0, 1.0, 1.0),
                   center=(0.0, 0.0, 0.0), rings=6, segments=10,
                   gloss=0.0, translucency=0.0, emissive=0.0) -> Mesh:
    """Low-poly smooth ellipsoid used for organic foliage and terrain."""
    rx, ry, rz = radii
    cx, cy, cz = center
    rings = max(3, int(rings))
    segments = max(5, int(segments))
    verts = []
    idx = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        sp, cp = math.sin(phi), math.cos(phi)
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            ct, st = math.cos(theta), math.sin(theta)
            ux, uy, uz = sp * ct, cp, sp * st
            x, y, z = cx + ux * rx, cy + uy * ry, cz + uz * rz
            # Inverse-transpose scale gives the correct smooth normal.
            nx, ny, nz = ux / max(rx, 1e-5), uy / max(ry, 1e-5), uz / max(rz, 1e-5)
            nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            verts.append([x, y, z, nx / nl, ny / nl, nz / nl, *color,
                          emissive, gloss, translucency])
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + s
            d = (r + 1) * segments + (s + 1) % segments
            idx += [a, c, d, a, d, b]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def _loop_perps(points):
    """Unit XZ-perpendicular at each vertex of a closed polyline (list of vec3-likes)."""
    import glm

    n = len(points)
    perps = []
    for i in range(n):
        prev = points[(i - 1) % n]
        cur = points[i]
        nxt = points[(i + 1) % n]
        din = glm.normalize(cur - prev) if glm.length(cur - prev) > 0 else glm.vec3(1, 0, 0)
        dout = glm.normalize(nxt - cur) if glm.length(nxt - cur) > 0 else din
        tangent = din + dout
        if glm.length(tangent) < 1e-6:
            tangent = dout
        tangent = glm.normalize(tangent)
        perp = glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x))
        perps.append(perp)
    return perps


def make_loop_ribbon(points, half_width, y, color, gloss=0.0) -> Mesh:
    """A flat ribbon (normal +Y) of the given half-width following a closed loop.

    ``points`` is a list of glm.vec3 (ground positions). Used for the track bed
    and the two rails.
    """
    perps = _loop_perps(points)
    n = len(points)
    up = (0.0, 1.0, 0.0)
    verts = []
    for i in range(n):
        p = points[i]
        perp = perps[i]
        left = p + perp * half_width
        right = p - perp * half_width
        verts.append([left.x, left.y + y, left.z, *up, *color, 0.0, gloss, 0.0])
        verts.append([right.x, right.y + y, right.z, *up, *color, 0.0, gloss, 0.0])
    idx = []
    for i in range(n):
        a = 2 * i          # left_i
        b = 2 * i + 1      # right_i
        c = 2 * ((i + 1) % n)      # left_{i+1}
        d = 2 * ((i + 1) % n) + 1  # right_{i+1}
        idx += [a, b, d, a, d, c]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_path_ribbon(points, half_width, y, color, gloss=0.0) -> Mesh:
    """Flat ribbon following an open path, with stable endpoint tangents."""
    import glm
    n = len(points)
    if n < 2:
        raise ValueError("A path ribbon needs at least two points")
    verts = []
    up = (0.0, 1.0, 0.0)
    for i, p in enumerate(points):
        prev = points[max(0, i - 1)]
        nxt = points[min(n - 1, i + 1)]
        tangent = glm.normalize(nxt - prev)
        perp = glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x))
        left, right = p + perp * half_width, p - perp * half_width
        verts.append([left.x, left.y + y, left.z, *up, *color, 0.0, gloss, 0.0])
        verts.append([right.x, right.y + y, right.z, *up, *color, 0.0, gloss, 0.0])
    idx = []
    for i in range(n - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1
        idx += [a, b, d, a, d, c]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_extruded_ribbon(points, half_width, bottom_y, top_y, color,
                         gloss=0.0, closed=True) -> Mesh:
    """A solid ribbon with top, bottom and side faces.

    This is the basic building block for real rail profiles: unlike the legacy
    flat ribbon it has visible height, silhouette, side shading and specular
    highlights on a separately authored rail head.
    """
    import glm

    n = len(points)
    if n < 2:
        raise ValueError("An extruded ribbon needs at least two points")
    if top_y <= bottom_y:
        raise ValueError("top_y must be above bottom_y")

    if closed:
        perps = _loop_perps(points)
    else:
        perps = []
        for i, p in enumerate(points):
            prev = points[max(0, i - 1)]
            nxt = points[min(n - 1, i + 1)]
            tangent = glm.normalize(nxt - prev)
            perps.append(glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x)))

    verts = []
    idx = []
    segment_count = n if closed else n - 1

    def quad(corners, normal):
        base = len(verts)
        for p in corners:
            verts.append([p.x, p.y, p.z, normal.x, normal.y, normal.z,
                          *color, 0.0, gloss, 0.0])
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    for i in range(segment_count):
        j = (i + 1) % n
        li = points[i] + perps[i] * half_width
        ri = points[i] - perps[i] * half_width
        lj = points[j] + perps[j] * half_width
        rj = points[j] - perps[j] * half_width
        lit, rit = glm.vec3(li.x, li.y + top_y, li.z), glm.vec3(ri.x, ri.y + top_y, ri.z)
        ljt, rjt = glm.vec3(lj.x, lj.y + top_y, lj.z), glm.vec3(rj.x, rj.y + top_y, rj.z)
        lib, rib = glm.vec3(li.x, li.y + bottom_y, li.z), glm.vec3(ri.x, ri.y + bottom_y, ri.z)
        ljb, rjb = glm.vec3(lj.x, lj.y + bottom_y, lj.z), glm.vec3(rj.x, rj.y + bottom_y, rj.z)
        side_n = glm.normalize(perps[i] + perps[j])
        quad([lit, rit, rjt, ljt], glm.vec3(0, 1, 0))
        quad([lib, ljb, rjb, rib], glm.vec3(0, -1, 0))
        quad([lib, lit, ljt, ljb], side_n)
        quad([rib, rjb, rjt, rit], -side_n)

    if not closed:
        quad([
            glm.vec3(points[0].x + perps[0].x * half_width, points[0].y + bottom_y,
                     points[0].z + perps[0].z * half_width),
            glm.vec3(points[0].x - perps[0].x * half_width, points[0].y + bottom_y,
                     points[0].z - perps[0].z * half_width),
            glm.vec3(points[0].x - perps[0].x * half_width, points[0].y + top_y,
                     points[0].z - perps[0].z * half_width),
            glm.vec3(points[0].x + perps[0].x * half_width, points[0].y + top_y,
                     points[0].z + perps[0].z * half_width),
        ], -glm.normalize(points[1] - points[0]))
        last = n - 1
        quad([
            glm.vec3(points[last].x - perps[last].x * half_width, points[last].y + bottom_y,
                     points[last].z - perps[last].z * half_width),
            glm.vec3(points[last].x + perps[last].x * half_width, points[last].y + bottom_y,
                     points[last].z + perps[last].z * half_width),
            glm.vec3(points[last].x + perps[last].x * half_width, points[last].y + top_y,
                     points[last].z + perps[last].z * half_width),
            glm.vec3(points[last].x - perps[last].x * half_width, points[last].y + top_y,
                     points[last].z - perps[last].z * half_width),
        ], glm.normalize(points[last] - points[last - 1]))

    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_oriented_box(size, color, center, forward, emissive=0.0,
                      gloss=0.0, translucency=0.0) -> Mesh:
    """Box baked into world space with local +X aligned to ``forward``."""
    import glm

    result = make_box(size=size, color=color, center=(0.0, 0.0, 0.0),
                      emissive=emissive, gloss=gloss,
                      translucency=translucency)
    fwd = glm.normalize(glm.vec3(forward.x, 0.0, forward.z))
    side = glm.vec3(-fwd.z, 0.0, fwd.x)
    cx, cy, cz = center
    for vertex in result.vertices:
        x, y, z = vertex[0:3]
        nx, ny, nz = vertex[3:6]
        p = fwd * float(x) + glm.vec3(0, 1, 0) * float(y) + side * float(z)
        normal = fwd * float(nx) + glm.vec3(0, 1, 0) * float(ny) + side * float(nz)
        vertex[0:3] = (p.x + cx, p.y + cy, p.z + cz)
        vertex[3:6] = (normal.x, normal.y, normal.z)
    return result


def make_rounded_box_x(length, height, width, color, center=(0.0, 0.0, 0.0),
                       chamfer=0.28, gloss=0.0) -> Mesh:
    """Extruded eight-sided body with chamfered roof and lower corners.

    The long axis is local X.  It gives rail vehicles a curved-looking body
    silhouette while remaining inexpensive enough for multiple cars.
    """
    cx, cy, cz = center
    hx, hy, hz = length / 2.0, height / 2.0, width / 2.0
    c = min(chamfer, hy * 0.45, hz * 0.45)
    # Counter-clockwise cross-section in the YZ plane.
    section = [
        (-hy + c, -hz), (hy - c, -hz), (hy, -hz + c), (hy, hz - c),
        (hy - c, hz), (-hy + c, hz), (-hy, hz - c), (-hy, -hz + c),
    ]
    verts = []
    idx = []

    for i in range(len(section)):
        j = (i + 1) % len(section)
        y0, z0 = section[i]
        y1, z1 = section[j]
        dy, dz = y1 - y0, z1 - z0
        nl = math.hypot(dy, dz) or 1.0
        ny, nz = dz / nl, -dy / nl
        base = len(verts)
        for x, y, z in [(-hx, y0, z0), (hx, y0, z0), (hx, y1, z1), (-hx, y1, z1)]:
            verts.append([cx + x, cy + y, cz + z, 0.0, ny, nz,
                          *color, 0.0, gloss, 0.0])
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]

    for x, nx, reverse in ((-hx, -1.0, True), (hx, 1.0, False)):
        base = len(verts)
        for y, z in section:
            verts.append([cx + x, cy + y, cz + z, nx, 0.0, 0.0,
                          *color, 0.0, gloss, 0.0])
        for i in range(1, len(section) - 1):
            tri = [base, base + i, base + i + 1]
            idx += list(reversed(tri)) if reverse else tri

    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_tapered_rounded_x(length, height, width, color, center=(0.0, 0.0, 0.0),
                           narrow_end=1, nose_scale=0.68, chamfer=0.28,
                           gloss=0.0) -> Mesh:
    """Chamfered transition whose selected X end narrows into a train nose."""
    cx, cy, cz = center
    hx, hy, hz = length / 2.0, height / 2.0, width / 2.0
    c = min(chamfer, hy * 0.42, hz * 0.42)
    base = [
        (-hy + c, -hz), (hy - c, -hz), (hy, -hz + c), (hy, hz - c),
        (hy - c, hz), (-hy + c, hz), (-hy, hz - c), (-hy, -hz + c),
    ]
    scales = ((nose_scale, 1.0) if narrow_end < 0 else (1.0, nose_scale))
    rings = []
    for x, scale in ((-hx, scales[0]), (hx, scales[1])):
        rings.append([np.array((cx + x, cy + y * scale, cz + z * scale), dtype=np.float32)
                      for y, z in base])
    verts, idx = [], []

    def quad(points):
        p0, p1, _, p3 = points
        normal = np.cross(p1 - p0, p3 - p0)
        norm = float(np.linalg.norm(normal)) or 1.0
        normal /= norm
        start = len(verts)
        for p in points:
            verts.append([*p.tolist(), *normal.tolist(), *color, 0.0, gloss, 0.0])
        idx.extend([start, start + 1, start + 2, start, start + 2, start + 3])

    for i in range(8):
        j = (i + 1) % 8
        quad([rings[0][i], rings[1][i], rings[1][j], rings[0][j]])
    for ring, nx, reverse in ((rings[0], -1.0, True), (rings[1], 1.0, False)):
        start = len(verts)
        for p in ring:
            verts.append([*p.tolist(), nx, 0.0, 0.0, *color, 0.0, gloss, 0.0])
        for i in range(1, 7):
            tri = [start, start + i, start + i + 1]
            idx.extend(reversed(tri) if reverse else tri)
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def _superellipse_section(height, width, segments=24, exponent=4.0):
    """Return a smooth, railcar-like YZ cross-section and analytic normals."""
    hy, hz = height * 0.5, width * 0.5
    power = 2.0 / exponent
    section = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cs, sn = math.cos(theta), math.sin(theta)
        y = hy * math.copysign(abs(cs) ** power, cs)
        z = hz * math.copysign(abs(sn) ** power, sn)
        # Gradient of |y/hy|^n + |z/hz|^n = 1.
        gy = math.copysign((abs(y / hy) ** (exponent - 1.0)) / hy, y)
        gz = math.copysign((abs(z / hz) ** (exponent - 1.0)) / hz, z)
        norm = math.hypot(gy, gz) or 1.0
        section.append((y, z, gy / norm, gz / norm))
    return section


def make_superellipse_body_x(length, height, width, color, center=(0.0, 0.0, 0.0),
                             segments=24, exponent=4.0, gloss=0.0) -> Mesh:
    """Smooth aluminium railcar shell with a rounded-square cross-section."""
    cx, cy, cz = center
    hx = length * 0.5
    section = _superellipse_section(height, width, segments, exponent)
    verts, idx = [], []
    for x in (-hx, hx):
        for y, z, ny, nz in section:
            verts.append([cx + x, cy + y, cz + z, 0.0, ny, nz,
                          *color, 0.0, gloss, 0.0])
    for i in range(segments):
        j = (i + 1) % segments
        idx += [i, segments + i, segments + j, i, segments + j, j]
    for ring, nx, reverse in ((0, -1.0, True), (segments, 1.0, False)):
        centre_idx = len(verts)
        verts.append([cx + (-hx if ring == 0 else hx), cy, cz,
                      nx, 0.0, 0.0, *color, 0.0, gloss, 0.0])
        for i in range(segments):
            j = (i + 1) % segments
            tri = [centre_idx, ring + i, ring + j]
            idx += list(reversed(tri)) if reverse else tri
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_streamlined_nose_x(base_x, end_sign, length, height, width, color,
                            base_y=0.0, segments=24, rings=7, exponent=3.6,
                            gloss=0.0) -> Mesh:
    """Multi-ring aerodynamic cab nose with a lowered, narrowed front mask."""
    base_section = _superellipse_section(height, width, segments, exponent)
    verts, idx = [], []
    for ring in range(rings):
        t = ring / (rings - 1)
        eased = t * t * (3.0 - 2.0 * t)
        width_scale = 1.0 - 0.40 * eased
        height_scale = 1.0 - 0.20 * eased
        y_shift = -0.25 * eased
        x = base_x + end_sign * length * t
        for y, z, ny, nz in base_section:
            # A small longitudinal normal component softens ring transitions.
            nx = end_sign * (0.12 + 0.32 * t)
            n = np.array((nx, ny / max(height_scale, 0.01),
                          nz / max(width_scale, 0.01)), dtype=np.float32)
            n /= float(np.linalg.norm(n)) or 1.0
            verts.append([x, base_y + y * height_scale + y_shift,
                          z * width_scale, *n.tolist(), *color,
                          0.0, gloss, 0.0])
    for ring in range(rings - 1):
        a0, b0 = ring * segments, (ring + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            if end_sign > 0:
                idx += [a0 + i, b0 + i, b0 + j, a0 + i, b0 + j, a0 + j]
            else:
                idx += [a0 + i, b0 + j, b0 + i, a0 + i, a0 + j, b0 + j]
    tip_ring = (rings - 1) * segments
    centre_idx = len(verts)
    verts.append([base_x + end_sign * length, base_y - 0.25, 0.0,
                  float(end_sign), 0.0, 0.0, *color, 0.0, gloss, 0.0])
    for i in range(segments):
        j = (i + 1) % segments
        if end_sign > 0:
            idx += [centre_idx, tip_ring + i, tip_ring + j]
        else:
            idx += [centre_idx, tip_ring + j, tip_ring + i]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_cylinder(radius, length, color, center=(0.0, 0.0, 0.0), axis="y",
                  segments=16, gloss=0.0, emissive=0.0) -> Mesh:
    """Smooth cylinder for wheels, columns, insulators and equipment."""
    cx, cy, cz = center
    half = length / 2.0
    verts, idx = [], []

    def map_pos(a, b, c):
        if axis == "x":
            return (cx + c, cy + a, cz + b)
        if axis == "z":
            return (cx + a, cy + b, cz + c)
        return (cx + a, cy + c, cz + b)

    def map_normal(a, b, c):
        if axis == "x":
            return (c, a, b)
        if axis == "z":
            return (a, b, c)
        return (a, c, b)

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        ca, sa = math.cos(angle), math.sin(angle)
        for end in (-half, half):
            p = map_pos(radius * ca, radius * sa, end)
            n = map_normal(ca, sa, 0.0)
            verts.append([*p, *n, *color, emissive, gloss, 0.0])
    for i in range(segments):
        j = (i + 1) % segments
        a, b, c, d = i * 2, i * 2 + 1, j * 2 + 1, j * 2
        idx += [a, b, c, a, c, d]
    for end_index, normal_sign in ((0, -1.0), (1, 1.0)):
        centre_index = len(verts)
        p = map_pos(0.0, 0.0, normal_sign * half)
        n = map_normal(0.0, 0.0, normal_sign)
        verts.append([*p, *n, *color, emissive, gloss, 0.0])
        for i in range(segments):
            j = (i + 1) % segments
            a, b = i * 2 + end_index, j * 2 + end_index
            idx += ([centre_index, b, a] if end_index == 0
                    else [centre_index, a, b])
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_tube_between(start, end, radius, color, segments=12, gloss=0.0,
                      emissive=0.0) -> Mesh:
    """Cylinder between arbitrary 3-D endpoints.

    This is used for pantograph linkages, handrails and diagonal structural
    members where axis-aligned boxes immediately betray procedural geometry.
    """
    a = np.asarray(start, dtype=np.float32)
    b = np.asarray(end, dtype=np.float32)
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length <= 1e-6:
        raise ValueError("Tube endpoints must not coincide")
    axis_y = delta / length
    reference = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    if abs(float(np.dot(reference, axis_y))) > 0.93:
        reference = np.array((1.0, 0.0, 0.0), dtype=np.float32)
    axis_x = np.cross(reference, axis_y)
    axis_x /= float(np.linalg.norm(axis_x))
    axis_z = np.cross(axis_y, axis_x)
    centre = (a + b) * 0.5
    result = make_cylinder(
        radius=radius, length=length, color=color, center=(0.0, 0.0, 0.0),
        axis="y", segments=segments, gloss=gloss, emissive=emissive,
    )
    for vertex in result.vertices:
        local = vertex[0:3].copy()
        normal = vertex[3:6].copy()
        world = centre + axis_x * local[0] + axis_y * local[1] + axis_z * local[2]
        world_normal = axis_x * normal[0] + axis_y * normal[1] + axis_z * normal[2]
        vertex[0:3] = world
        vertex[3:6] = world_normal
    return result


def make_rounded_panel(width, height, depth, color, center=(0.0, 0.0, 0.0),
                       normal_axis="z", corner_radius=0.12, corner_segments=3,
                       gloss=0.0, emissive=0.0, translucency=0.0) -> Mesh:
    """Thin rounded rectangle for glazing, doors, displays and light lenses.

    ``normal_axis`` selects the panel thickness direction: ``z`` for bodyside
    panels, ``x`` for cab-end panels and ``y`` for rounded building plans.  The
    perimeter has true curved corners rather than being a rectangular decal.
    """
    if normal_axis not in ("x", "y", "z"):
        raise ValueError("Rounded panel normal_axis must be 'x', 'y' or 'z'")
    half_w, half_h = width * 0.5, height * 0.5
    radius = min(corner_radius, half_w, half_h)
    perimeter = []
    for cu, cv, start_angle in (
        (half_w - radius, half_h - radius, 0.0),
        (-half_w + radius, half_h - radius, math.pi * 0.5),
        (-half_w + radius, -half_h + radius, math.pi),
        (half_w - radius, -half_h + radius, math.pi * 1.5),
    ):
        for step in range(corner_segments + 1):
            angle = start_angle + math.pi * 0.5 * step / corner_segments
            perimeter.append((cu + math.cos(angle) * radius,
                              cv + math.sin(angle) * radius))

    cx, cy, cz = center
    half_d = depth * 0.5

    def map_pos(u, v, w):
        if normal_axis == "x":
            return (cx + w, cy + v, cz + u)
        if normal_axis == "y":
            return (cx + u, cy + w, cz + v)
        return (cx + u, cy + v, cz + w)

    def map_normal(u, v, w):
        if normal_axis == "x":
            return (w, v, u)
        if normal_axis == "y":
            return (u, w, v)
        return (u, v, w)

    verts, idx = [], []
    count = len(perimeter)
    # Smooth perimeter wall.
    for i, (u, v) in enumerate(perimeter):
        length_uv = math.hypot(u, v) or 1.0
        nu, nv = u / length_uv, v / length_uv
        for w in (-half_d, half_d):
            p = map_pos(u, v, w)
            n = map_normal(nu, nv, 0.0)
            verts.append([*p, *n, *color, emissive, gloss, translucency])
    for i in range(count):
        j = (i + 1) % count
        a, b, c, d = i * 2, j * 2, j * 2 + 1, i * 2 + 1
        idx += [a, b, c, a, c, d]
    # Front and back faces, triangulated as fans.
    for side, sign in ((0, -1.0), (1, 1.0)):
        centre_idx = len(verts)
        p = map_pos(0.0, 0.0, sign * half_d)
        n = map_normal(0.0, 0.0, sign)
        verts.append([*p, *n, *color, emissive, gloss, translucency])
        for u, v in perimeter:
            p = map_pos(u, v, sign * half_d)
            verts.append([*p, *n, *color, emissive, gloss, translucency])
        for i in range(count):
            j = (i + 1) % count
            if side == 0:
                idx += [centre_idx, centre_idx + 1 + j, centre_idx + 1 + i]
            else:
                idx += [centre_idx, centre_idx + 1 + i, centre_idx + 1 + j]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_tunnel_shell(points, horizontal_radius, vertical_radius, center_height,
                      color, radial_segments=20, gloss=0.0) -> Mesh:
    """Sweep an inward-facing elliptical concrete shell along an open route."""
    import glm

    if len(points) < 2:
        raise ValueError("Tunnel shell needs at least two route points")
    verts, idx = [], []
    for i, point in enumerate(points):
        prev = points[max(0, i - 1)]
        nxt = points[min(len(points) - 1, i + 1)]
        tangent = glm.normalize(nxt - prev)
        perp = glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x))
        for r in range(radial_segments):
            theta = 2.0 * math.pi * r / radial_segments
            cs, sn = math.cos(theta), math.sin(theta)
            p = point + perp * (cs * horizontal_radius)
            p.y += center_height + sn * vertical_radius
            outward = glm.normalize(perp * (cs / horizontal_radius)
                                    + glm.vec3(0, 1, 0) * (sn / vertical_radius))
            normal = -outward
            shade = 0.96 + 0.04 * math.sin(i * 1.7 + r * 0.9)
            col = tuple(max(0.0, min(1.0, c * shade)) for c in color)
            verts.append([p.x, p.y, p.z, normal.x, normal.y, normal.z,
                          *col, 0.0, gloss, 0.0])
    rings = len(points)
    for i in range(rings - 1):
        for r in range(radial_segments):
            q = (r + 1) % radial_segments
            a = i * radial_segments + r
            b = (i + 1) * radial_segments + r
            c = (i + 1) * radial_segments + q
            d = i * radial_segments + q
            idx += [a, c, b, a, d, c]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def offset_path(points, distance):
    """Shift an open path sideways by ``distance``."""
    import glm
    out = []
    for i, p in enumerate(points):
        prev = points[max(0, i - 1)]
        nxt = points[min(len(points) - 1, i + 1)]
        tangent = glm.normalize(nxt - prev)
        out.append(p + glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x)) * distance)
    return out


def offset_loop(points, distance):
    """Return points shifted along their per-vertex perpendicular by ``distance``."""
    perps = _loop_perps(points)
    return [points[i] + perps[i] * distance for i in range(len(points))]


def combine(meshes) -> Mesh:
    """Merge several meshes into one, offsetting indices."""
    all_verts = []
    all_idx = []
    offset = 0
    for m in meshes:
        all_verts.append(m.vertices)
        all_idx.append(m.indices + offset)
        offset += m.vertices.shape[0]
    return Mesh(
        np.concatenate(all_verts, axis=0).astype(np.float32),
        np.concatenate(all_idx, axis=0).astype(np.uint32),
    )
