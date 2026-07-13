"""Build the static world mesh (ground, track, stations) and the train mesh."""

from __future__ import annotations

import math
import random

import glm
import numpy as np

from .. import config
from . import mesh
from .mesh import Mesh


TRACK_BED_COLOR = (0.46, 0.41, 0.36)     # sunlit ballast gravel
BALLAST_SHOULDER = (0.29, 0.27, 0.25)
SLEEPER_COLOR = (0.26, 0.21, 0.17)
FASTENER_COLOR = (0.32, 0.33, 0.35)
RAIL_FOOT_COLOR = (0.40, 0.41, 0.43)
RAIL_WEB_COLOR = (0.47, 0.48, 0.50)
VERGE_COLOR = (0.34, 0.44, 0.21)         # grass strip beside the ballast
PLATFORM_COLOR = (0.62, 0.62, 0.66)
BUILDING_COLOR = (0.72, 0.70, 0.66)
MAST_COLOR = (0.38, 0.40, 0.42)
WIRE_COLOR = (0.07, 0.07, 0.08)

SCENERY_SEED = 4321
CATENARY_SPACING = 22.0   # metres of arc between masts
TREE_SPACING = 35.0       # realistic corridor landscaping interval
WIRE_HEIGHT = 6.2
SLEEPER_SPACING = 4.5     # performance-aware representation on the 59 km route
VISIBLE_SLEEPER_SPACING = 0.72  # real high-speed slab/ballast track rhythm
SIGNAL_POST_COLOR = (0.20, 0.22, 0.24)
SIGNAL_HEAD_COLOR = (0.055, 0.065, 0.075)


def _rail_profile(parts, centre_points, closed):
    """Add foot, web and polished head for both running rails."""
    offset = mesh.offset_loop if closed else mesh.offset_path
    for gauge in (-config.RAIL_HALF_WIDTH, config.RAIL_HALF_WIDTH):
        rail = offset(centre_points, gauge)
        parts.append(mesh.make_extruded_ribbon(
            rail, 0.19, 0.19, 0.25, RAIL_FOOT_COLOR, gloss=0.48, closed=closed,
        ))
        parts.append(mesh.make_extruded_ribbon(
            rail, 0.075, 0.25, 0.39, RAIL_WEB_COLOR, gloss=0.62, closed=closed,
        ))
        parts.append(mesh.make_extruded_ribbon(
            rail, 0.12, 0.39, 0.49, config.RAIL_COLOR, gloss=0.98, closed=closed,
        ))


def _sleeper(parts, pos, fwd, include_fasteners=True):
    """Concrete/wood sleeper across the track with four steel clips."""
    perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
    parts.append(mesh.make_oriented_box(
        size=(config.RAIL_HALF_WIDTH * 2.0 + 1.45, 0.17, 0.28),
        color=SLEEPER_COLOR, center=(pos.x, pos.y + 0.145, pos.z), forward=perp,
        gloss=0.12,
    ))
    if include_fasteners:
        for gauge in (-config.RAIL_HALF_WIDTH, config.RAIL_HALF_WIDTH):
            rail_c = pos + perp * gauge
            for side in (-0.17, 0.17):
                clip = rail_c + perp * side
                parts.append(mesh.make_oriented_box(
                    size=(0.12, 0.09, 0.16), color=FASTENER_COLOR,
                    center=(clip.x, clip.y + 0.235, clip.z), forward=perp, gloss=0.78,
                ))


def _dense_sleeper_tops(line, route_offset):
    """Batch the close-spaced sleeper top faces into one efficient mesh."""
    verts = []
    indices = []
    d = 0.0
    half_along = 0.16
    half_across = config.RAIL_HALF_WIDTH + 0.72
    while d < line.total:
        pos, fwd = line.position_at(d)
        flat_fwd = glm.normalize(glm.vec3(fwd.x, 0.0, fwd.z))
        perp = glm.normalize(glm.vec3(-flat_fwd.z, 0.0, flat_fwd.x))
        pos += perp * route_offset
        a = flat_fwd * half_along
        b = perp * half_across
        y = pos.y + 0.245
        corners = (pos - a - b, pos + a - b, pos + a + b, pos - a + b)
        base = len(verts)
        for p in corners:
            verts.append([p.x, y, p.z, 0.0, 1.0, 0.0,
                          *SLEEPER_COLOR, 0.0, 0.14, 0.0])
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
        d += VISIBLE_SLEEPER_SPACING
    return Mesh(np.asarray(verts, dtype=np.float32),
                np.asarray(indices, dtype=np.uint32))


def _loop_track(parts, line, route_offset):
    points = (line.points if route_offset == 0.0
              else mesh.offset_path(line.points, route_offset))
    parts.append(mesh.make_path_ribbon(points, 3.25, 0.035, BALLAST_SHOULDER))
    parts.append(mesh.make_path_ribbon(
        points, config.RAIL_HALF_WIDTH + 0.95, 0.075, TRACK_BED_COLOR,
    ))
    _rail_profile(parts, points, closed=False)
    parts.append(_dense_sleeper_tops(line, route_offset))
    d, index = 0.0, 0
    while d < line.total:
        pos, fwd = line.position_at(d)
        perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
        pos += perp * route_offset
        _sleeper(parts, pos, fwd, include_fasteners=(index % 8 == 0))
        d += SLEEPER_SPACING
        index += 1


def _main_track(parts, line):
    """Two complete running tracks, including the AI opposite-direction line."""
    _loop_track(parts, line, 0.0)
    _loop_track(parts, line, config.OPPOSING_TRACK_OFFSET)


def _surface_verges(parts, line):
    """Add grass shoulders only where the alignment is actually above ground."""
    segment = []
    for point in line.points:
        if point.y >= -1.0:
            segment.append(point)
        else:
            if len(segment) >= 2:
                parts.append(mesh.make_path_ribbon(
                    segment, 7.5, 0.02, VERGE_COLOR, gloss=0.04,
                ))
            segment = []
    if len(segment) >= 2:
        parts.append(mesh.make_path_ribbon(
            segment, 7.5, 0.02, VERGE_COLOR, gloss=0.04,
        ))


def _path_pose(points, distance):
    remaining = distance
    for i in range(len(points) - 1):
        delta = points[i + 1] - points[i]
        length = glm.length(delta)
        if remaining <= length or i == len(points) - 2:
            fwd = glm.normalize(delta)
            t = 0.0 if length <= 1e-6 else min(1.0, remaining / length)
            return points[i] + delta * t, fwd
        remaining -= length
    return points[-1], glm.vec3(1, 0, 0)


def _branch_track(parts, points):
    parts.append(mesh.make_path_ribbon(points, 3.0, 0.036, BALLAST_SHOULDER))
    parts.append(mesh.make_path_ribbon(
        points, config.RAIL_HALF_WIDTH + 0.95, 0.076, TRACK_BED_COLOR,
    ))
    _rail_profile(parts, points, closed=False)
    total = sum(glm.length(points[i + 1] - points[i]) for i in range(len(points) - 1))
    d, index = 0.4, 0
    while d < total:
        pos, fwd = _path_pose(points, d)
        _sleeper(parts, pos, fwd, include_fasteners=(index % 2 == 0))
        d += SLEEPER_SPACING
        index += 1


def _turnout_detail(parts, points):
    """Model the working parts that visually identify a real turnout."""
    if len(points) < 2:
        return
    toe = points[0]
    branch_fwd = glm.normalize(points[1] - points[0])
    branch_perp = glm.normalize(glm.vec3(-branch_fwd.z, 0.0, branch_fwd.x))

    # Long switch timbers, tapered point blades and the two check rails around
    # the crossing.  They sit slightly above the ordinary permanent way.
    for along in range(0, 19, 2):
        p = toe + branch_fwd * float(along)
        parts.append(mesh.make_oriented_box(
            size=(config.RAIL_HALF_WIDTH * 2.0 + 2.0, 0.18, 0.34),
            color=(0.22, 0.18, 0.14), center=(p.x, p.y + 0.14, p.z),
            forward=branch_perp, gloss=0.10,
        ))
    for side in (-1.0, 1.0):
        blade_start = toe + branch_perp * side * (config.RAIL_HALF_WIDTH - 0.06)
        blade_end = toe + branch_fwd * 15.0 + branch_perp * side * config.RAIL_HALF_WIDTH
        parts.append(mesh.make_tube_between(
            (blade_start.x, blade_start.y + 0.44, blade_start.z),
            (blade_end.x, blade_end.y + 0.46, blade_end.z),
            0.055, (0.60, 0.61, 0.62), segments=10, gloss=0.96,
        ))
        guard_a = toe + branch_fwd * 10.0 + branch_perp * side * (config.RAIL_HALF_WIDTH - 0.26)
        guard_b = toe + branch_fwd * 18.0 + branch_perp * side * (config.RAIL_HALF_WIDTH - 0.26)
        parts.append(mesh.make_tube_between(
            (guard_a.x, guard_a.y + 0.43, guard_a.z),
            (guard_b.x, guard_b.y + 0.43, guard_b.z),
            0.045, (0.38, 0.39, 0.40), segments=8, gloss=0.78,
        ))

    # Electric point machine, throw bar and reflective position marker.
    motor = toe + branch_fwd * 3.6 + branch_perp * 2.45
    parts.append(mesh.make_rounded_box_x(
        1.15, 0.42, 0.72, (0.20, 0.23, 0.24),
        center=(motor.x, motor.y + 0.43, motor.z), chamfer=0.12, gloss=0.52,
    ))
    rod_end = toe + branch_fwd * 3.6 - branch_perp * 1.5
    parts.append(mesh.make_tube_between(
        (motor.x, motor.y + 0.36, motor.z),
        (rod_end.x, rod_end.y + 0.36, rod_end.z),
        0.035, (0.66, 0.67, 0.68), segments=8, gloss=0.86,
    ))
    marker = motor + glm.vec3(0.0, 0.65, 0.0)
    parts.append(mesh.make_rounded_panel(
        0.34, 0.34, 0.07, (0.90, 0.76, 0.10),
        center=(marker.x, marker.y, marker.z), normal_axis="z",
        corner_radius=0.06, emissive=0.22, gloss=0.50,
    ))


def _route_crossovers(parts, line):
    """Scissors/crossover routes for regulation and terminal turnback moves."""
    # Crossovers at real station-throat scale, placed clear of the platforms.
    for station_index in (1, 2, 4, 5):
        start_d = line.stations[station_index].distance + 330.0
        crossover_length = 118.0
        points = []
        for sample in range(13):
            t = sample / 12.0
            smooth = t * t * (3.0 - 2.0 * t)
            pos, fwd = line.position_at(start_d + crossover_length * t)
            perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
            points.append(pos + perp * (config.OPPOSING_TRACK_OFFSET * smooth))
        _branch_track(parts, points)
        _turnout_detail(parts, points)
        _turnout_detail(parts, list(reversed(points)))

    # A centre pocket/turnback track beyond Pudong terminal supports regulation
    # moves without blocking either running line.
    terminal_d = line.stations[-1].distance
    pocket_points = []
    for sample in range(19):
        d = terminal_d - 720.0 + sample * 50.0
        pos, fwd = line.position_at(d)
        perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
        pocket_points.append(pos + perp * (config.OPPOSING_TRACK_OFFSET * 0.5))
    _branch_track(parts, pocket_points)
    _turnout_detail(parts, pocket_points)
    _turnout_detail(parts, list(reversed(pocket_points)))


def _tree(parts, x, z, rng):
    """Layered organic tree with trunk, branches and irregular leaf clusters."""
    trunk_h = rng.uniform(1.0, 1.6)
    h = rng.uniform(2.8, 4.6)
    g = rng.uniform(-0.04, 0.04)
    green = (0.16 + g, 0.34 + rng.uniform(-0.04, 0.08), 0.11 + g)
    parts.append(mesh.make_cylinder(
        radius=0.18, length=trunk_h, axis="y", segments=9,
        color=(0.28, 0.20, 0.12), center=(x, trunk_h / 2, z), gloss=0.08,
    ))
    w = rng.uniform(1.8, 3.0)
    # A few visible branches break the lollipop silhouette.
    for angle in (0.0, 2.1, 4.2):
        bx, bz = x + glm.cos(angle) * 0.35, z + glm.sin(angle) * 0.35
        parts.append(mesh.make_tube_between(
            (x, trunk_h * 0.74, z),
            (bx, trunk_h + h * 0.38, bz), radius=0.07,
            color=(0.25, 0.17, 0.10), segments=7, gloss=0.06,
        ))
    # Overlapping ellipsoids provide rounded, non-repeating crowns.  Smooth
    # normals plus translucency produce believable leaf clumps under backlight.
    clusters = [
        (-0.32, 0.24, 0.10, 0.72), (0.34, 0.28, -0.18, 0.68),
        (0.05, 0.50, 0.30, 0.62), (-0.08, 0.67, -0.12, 0.55),
    ]
    for ci, (ox, oy, oz, scale) in enumerate(clusters):
        tint = 0.88 + ci * 0.055 + rng.uniform(-0.04, 0.04)
        col = tuple(max(0.02, min(1.0, c * tint)) for c in green)
        parts.append(mesh.make_ellipsoid(
            radii=(w * scale, h * (0.23 + 0.025 * ci), w * scale * 0.82),
            color=col,
            center=(x + ox * w, trunk_h + oy * h, z + oz * w),
            rings=5, segments=8, translucency=0.78,
        ))


def _near_station_arc(line, d, margin):
    """True if arc-distance ``d`` lies within ``margin`` of any station node."""
    for s in line.stations:
        delta = abs(d - s.distance)
        if delta <= margin:
            return True
    return False


def _catenary(parts, line):
    """Portal-style masts and contact wires serving both running lines."""
    d = 0.0
    mast_index = 0
    while d < line.total:
        pos, fwd = line.position_at(d)
        perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
        if pos.y < -1.0:
            d += CATENARY_SPACING
            continue
        outer = config.OPPOSING_TRACK_OFFSET - config.RAIL_HALF_WIDTH - 2.1
        base = pos + perp * outer
        parts.append(mesh.make_cylinder(
            radius=0.13, length=WIRE_HEIGHT + 0.72, axis="y", segments=12,
            color=MAST_COLOR,
            center=(base.x, pos.y + (WIRE_HEIGHT + 0.72) / 2, base.z), gloss=0.48,
        ))
        # Triangulated cantilever, registration arms and brown ceramic
        # insulators match the line-side equipment visible in the references.
        a = glm.vec3(base.x, pos.y + WIRE_HEIGHT + 0.25, base.z)
        far = pos + perp * (config.RAIL_HALF_WIDTH + 2.1)
        b = glm.vec3(far.x, pos.y + WIRE_HEIGHT + 0.25, far.z)
        brace_base = glm.vec3(base.x, pos.y + WIRE_HEIGHT - 0.75, base.z)
        parts.append(mesh.make_tube_between(
            (a.x, a.y, a.z), (b.x, b.y, b.z), 0.045,
            MAST_COLOR, segments=9, gloss=0.56,
        ))
        parts.append(mesh.make_tube_between(
            (brace_base.x, brace_base.y, brace_base.z), (b.x, b.y, b.z), 0.040,
            MAST_COLOR, segments=9, gloss=0.52,
        ))
        for track_offset in (0.0, config.OPPOSING_TRACK_OFFSET):
            ins = pos + perp * track_offset
            parts.append(mesh.make_cylinder(
                radius=0.08, length=0.42, axis="y", segments=10,
                color=(0.40, 0.22, 0.11),
                center=(ins.x, pos.y + WIRE_HEIGHT + 0.34, ins.z), gloss=0.68,
            ))
            if mast_index % 2 == 0:
                parts.append(mesh.make_tube_between(
                    (ins.x, pos.y + WIRE_HEIGHT + 0.52, ins.z),
                    (ins.x, pos.y + WIRE_HEIGHT + 0.04, ins.z),
                    0.016, WIRE_COLOR, segments=6, gloss=0.68,
                ))
        d += CATENARY_SPACING
        mast_index += 1

    # Separate messenger/contact wires over both tracks.
    parts.append(mesh.make_path_ribbon(line.points, 0.027, WIRE_HEIGHT, WIRE_COLOR, gloss=0.68))
    parts.append(mesh.make_path_ribbon(line.points, 0.020, WIRE_HEIGHT + 0.52, WIRE_COLOR, gloss=0.62))
    opposing = mesh.offset_path(line.points, config.OPPOSING_TRACK_OFFSET)
    parts.append(mesh.make_path_ribbon(opposing, 0.027, WIRE_HEIGHT, WIRE_COLOR, gloss=0.68))
    parts.append(mesh.make_path_ribbon(opposing, 0.020, WIRE_HEIGHT + 0.52, WIRE_COLOR, gloss=0.62))


def signal_sites(line):
    """Return signal metadata shared by static housings and live lamps."""
    sites = []
    d = config.SIGNAL_SPACING * 0.45
    while d < line.total:
        pos, fwd = line.position_at(d)
        perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
        main_post = pos + perp * (config.RAIL_HALF_WIDTH + 1.45)
        opp_centre = pos + perp * config.OPPOSING_TRACK_OFFSET
        opp_post = opp_centre - perp * (config.RAIL_HALF_WIDTH + 1.45)
        sites.append({"track": "main", "distance": d,
                      "post": main_post, "direction": fwd})
        sites.append({"track": "opposing", "distance": d,
                      "post": opp_post, "direction": -fwd})
        d += config.SIGNAL_SPACING
    return sites


def _signal_housings(parts, line):
    for site in signal_sites(line):
        p, direction = site["post"], site["direction"]
        parts.append(mesh.make_cylinder(
            radius=0.28, length=0.30, axis="y", segments=14,
            color=(0.42, 0.43, 0.42),
            center=(p.x, p.y + 0.15, p.z), gloss=0.20,
        ))
        parts.append(mesh.make_cylinder(
            radius=0.095, length=3.65, axis="y", segments=12,
            color=SIGNAL_POST_COLOR,
            center=(p.x, p.y + 1.825, p.z), gloss=0.52,
        ))
        parts.append(mesh.make_oriented_box(
            size=(0.30, 1.42, 0.72), color=SIGNAL_HEAD_COLOR,
            center=(p.x, p.y + 3.72, p.z), forward=direction, gloss=0.25,
        ))
        # Three recessed lens barrels/visors make the aspects readable even
        # when a lamp is dark; the live emissive lens is drawn inside them.
        for aspect_y in (3.34, 3.72, 4.10):
            lens_c = p - direction * 0.24 + glm.vec3(0.0, aspect_y, 0.0)
            parts.append(mesh.make_tube_between(
                (lens_c.x - direction.x * 0.16, lens_c.y,
                 lens_c.z - direction.z * 0.16),
                (lens_c.x + direction.x * 0.16, lens_c.y,
                 lens_c.z + direction.z * 0.16),
                0.205, (0.025, 0.030, 0.034), segments=14, gloss=0.36,
            ))
        # Maintenance ladder on the back of the mast.
        ladder_side = glm.normalize(glm.vec3(-direction.z, 0.0, direction.x))
        for side in (-1.0, 1.0):
            rail = p + direction * 0.12 + ladder_side * side * 0.18
            parts.append(mesh.make_tube_between(
                (rail.x, p.y + 0.45, rail.z), (rail.x, p.y + 3.15, rail.z),
                0.025, (0.31, 0.33, 0.34), segments=7, gloss=0.52,
            ))
        for rung_y in (0.65, 1.05, 1.45, 1.85, 2.25, 2.65, 3.05):
            rung_c = p + direction * 0.12
            parts.append(mesh.make_tube_between(
                (rung_c.x - ladder_side.x * 0.18, p.y + rung_y,
                 rung_c.z - ladder_side.z * 0.18),
                (rung_c.x + ladder_side.x * 0.18, p.y + rung_y,
                 rung_c.z + ladder_side.z * 0.18),
                0.020, (0.31, 0.33, 0.34), segments=7, gloss=0.52,
            ))
        # White identification board beneath the head.
        board = p - direction * 0.11
        parts.append(mesh.make_oriented_box(
            size=(0.12, 0.38, 0.55), color=(0.76, 0.78, 0.74),
            center=(board.x, p.y + 2.75, board.z), forward=direction, gloss=0.18,
        ))


def build_signal_lamp_mesh(color) -> Mesh:
    """Small emissive lens instanced at the active aspect position."""
    return mesh.make_ellipsoid(
        radii=(0.16, 0.19, 0.16), color=color, center=(0.0, 0.0, 0.0),
        rings=5, segments=10, gloss=0.75, emissive=1.0,
    )


def _lineside_trees(parts, line, rng):
    """Greenery along the corridor, clear of platforms and buildings."""
    d = rng.uniform(0.0, TREE_SPACING)
    while d < line.total:
        step = rng.uniform(TREE_SPACING * 0.6, TREE_SPACING * 1.6)
        if not _near_station_arc(line, d, 14.0):
            pos, fwd = line.position_at(d)
            if pos.y < -1.0:
                d += step
                continue
            perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
            side = 1.0 if rng.random() < 0.5 else -1.0
            off = rng.uniform(9.0, 15.0)
            p = pos + perp * (side * off)
            _tree(parts, p.x + rng.uniform(-1.5, 1.5), p.z + rng.uniform(-1.5, 1.5), rng)
        d += step


def _underground_station(parts, line, station, variant=0):
    """Enclosed Airport Link Line cavern with platform screen doors."""
    node, tangent = line.position_at(station.distance)
    tangent = glm.normalize(tangent)
    perp = glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x))
    is_long = ("虹桥" in station.name or "旅游度假区" in station.name
               or "浦东" in station.name)
    length = config.LONG_PLATFORM_LENGTH if is_long else config.STATION_SIZE[0]
    platform_w = 5.6
    platform_off = config.TRAIN_SIZE[2] / 2.0 + 0.18 + platform_w / 2.0
    centres = [
        (node + perp * platform_off, -1.0),
        (node + perp * (config.OPPOSING_TRACK_OFFSET - platform_off), 1.0),
    ]
    corridor_mid = node + perp * (config.OPPOSING_TRACK_OFFSET * 0.5)

    # Cavern floor, ceiling and bright architectural side walls.
    parts.append(mesh.make_oriented_box(
        size=(length + 8.0, 0.35, 20.0), color=(0.32, 0.34, 0.36),
        center=(corridor_mid.x, node.y - 0.02, corridor_mid.z),
        forward=tangent, gloss=0.16,
    ))
    parts.append(mesh.make_oriented_box(
        size=(length + 8.0, 0.45, 20.0), color=(0.66, 0.68, 0.70),
        center=(corridor_mid.x, node.y + 6.8, corridor_mid.z),
        forward=tangent, gloss=0.24,
    ))
    for side in (-1.0, 1.0):
        wall = corridor_mid + perp * side * 9.8
        parts.append(mesh.make_oriented_box(
            size=(length + 8.0, 6.8, 0.40), color=(0.58, 0.60, 0.62),
            center=(wall.x, node.y + 3.35, wall.z), forward=tangent, gloss=0.20,
        ))

    for platform_c, edge_sign in centres:
        parts.append(mesh.make_oriented_box(
            size=(length, 0.58, platform_w), color=(0.68, 0.69, 0.70),
            center=(platform_c.x, node.y + 0.35, platform_c.z),
            forward=tangent, gloss=0.20,
        ))
        edge = platform_c + perp * edge_sign * (platform_w / 2.0 - 0.18)
        parts.append(mesh.make_oriented_box(
            size=(length * 0.98, 0.04, 0.32), color=(0.94, 0.77, 0.12),
            center=(edge.x, node.y + 0.66, edge.z), forward=tangent, gloss=0.25,
        ))
        # Full-height glass screen doors, dark frames and blue route fascia.
        for along in range(int(-length / 2 + 3), int(length / 2 - 2), 4):
            door = edge + tangent * float(along)
            parts.append(mesh.make_oriented_box(
                size=(2.55, 2.35, 0.10), color=(0.24, 0.39, 0.48),
                center=(door.x, node.y + 1.82, door.z), forward=tangent, gloss=0.96,
                translucency=0.62,
            ))
            for frame_offset in (-1.30, 1.30):
                frame = door + tangent * frame_offset
                parts.append(mesh.make_box(
                    size=(0.10, 2.55, 0.13), color=(0.055, 0.065, 0.075),
                    center=(frame.x, node.y + 1.88, frame.z), gloss=0.45,
                ))
        sign = platform_c - perp * edge_sign * 0.95
        parts.append(mesh.make_oriented_box(
            size=(12.0, 0.70, 0.14), color=station.color,
            center=(sign.x, node.y + 3.75, sign.z), forward=tangent,
            emissive=0.55, gloss=0.42,
        ))

        # Structural columns, brushed-metal bases and passenger information
        # displays give the five underground stations a believable rhythm.
        for along in range(int(-length / 2 + 10), int(length / 2 - 9), 18):
            column = platform_c + tangent * float(along) - perp * edge_sign * 1.25
            parts.append(mesh.make_cylinder(
                radius=0.24 if variant % 2 else 0.20, length=4.9, axis="y",
                segments=14, color=(0.46, 0.49, 0.51),
                center=(column.x, node.y + 3.10, column.z), gloss=0.62,
            ))
            parts.append(mesh.make_cylinder(
                radius=0.32, length=0.46, axis="y", segments=14,
                color=station.color, center=(column.x, node.y + 0.88, column.z),
                gloss=0.74,
            ))
        for along in (-length * 0.29, 0.0, length * 0.29):
            display = platform_c + tangent * along - perp * edge_sign * 0.78
            parts.append(mesh.make_rounded_panel(
                3.7, 0.70, 0.10, (0.025, 0.07, 0.10),
                center=(display.x, node.y + 4.35, display.z), normal_axis="z",
                corner_radius=0.12, emissive=0.36, gloss=0.92,
            ))

    # Continuous ceiling luminaires create a bright, modern airport station.
    for along in range(int(-length / 2 + 4), int(length / 2 - 3), 8):
        p = corridor_mid + tangent * float(along)
        for lateral in (-6.2, -3.2, 0.0, 3.2, 6.2):
            lamp = p + perp * lateral
            parts.append(mesh.make_oriented_box(
                size=(2.6, 0.12, 0.32), color=(0.92, 0.97, 1.0),
                center=(lamp.x, node.y + 6.48, lamp.z), forward=tangent,
                emissive=1.0, gloss=0.60,
            ))

    # Each underground station gets its own ceiling/wall language instead of a
    # single cloned box.  The repeated fins echo the real line's contemporary
    # transfer-hall architecture while retaining the station accent colour.
    fin_spacing = (5, 7, 6, 8, 5)[variant % 5]
    fin_color = tuple(min(1.0, c * 1.12 + 0.06) for c in station.color)
    for along in range(int(-length / 2 + 5), int(length / 2 - 4), fin_spacing):
        p = corridor_mid + tangent * float(along)
        parts.append(mesh.make_tube_between(
            (p.x - perp.x * 7.7, node.y + 6.25, p.z - perp.z * 7.7),
            (p.x + perp.x * 7.7, node.y + 6.25, p.z + perp.z * 7.7),
            radius=0.055 if variant != 3 else 0.075,
            color=fin_color, segments=8, gloss=0.48,
        ))
    for side in (-1.0, 1.0):
        wall_base = corridor_mid + perp * side * 9.53
        for along in range(int(-length / 2 + 6), int(length / 2 - 5), 12):
            panel = wall_base + tangent * float(along)
            parts.append(mesh.make_oriented_box(
                size=(8.8, 0.42, 0.08), color=fin_color,
                center=(panel.x, node.y + 2.85 + 0.22 * (variant % 3), panel.z),
                forward=tangent, emissive=0.16, gloss=0.56,
            ))

    _underground_signature(
        parts, node, tangent, perp, corridor_mid, centres,
        length, station.color, variant,
    )


def _platform_escalator(parts, base, tangent, perp, node_y, accent, lateral_sign):
    """Detailed inclined escalator with individual treads and handrails."""
    run, rise, steps = 13.5, 3.55, 18
    for index in range(steps):
        t = index / (steps - 1)
        p = base + tangent * (run * (t - 0.5))
        y = node_y + 0.72 + rise * t
        parts.append(mesh.make_oriented_box(
            size=(run / steps * 0.92, 0.10, 1.32),
            color=(0.24, 0.26, 0.27), center=(p.x, y, p.z),
            forward=tangent, gloss=0.58,
        ))
    low = base - tangent * (run * 0.5)
    high = base + tangent * (run * 0.5)
    for handrail_side in (-1.0, 1.0):
        rail_low = low + perp * handrail_side * 0.78
        rail_high = high + perp * handrail_side * 0.78
        parts.append(mesh.make_tube_between(
            (rail_low.x, node_y + 1.55, rail_low.z),
            (rail_high.x, node_y + rise + 1.55, rail_high.z),
            0.055, (0.07, 0.08, 0.09), segments=10, gloss=0.82,
        ))
        # Brushed-metal balustrade posts follow the incline.
        for marker in (0.15, 0.40, 0.65, 0.90):
            q = low + tangent * (run * marker) + perp * handrail_side * 0.78
            qy = node_y + 0.72 + rise * marker
            parts.append(mesh.make_tube_between(
                (q.x, qy, q.z), (q.x, qy + 0.82, q.z),
                0.025, (0.43, 0.46, 0.48), segments=8, gloss=0.70,
            ))
    top_landing = high + tangent * 1.5
    parts.append(mesh.make_rounded_panel(
        3.0, 2.7, 0.16, accent,
        center=(top_landing.x, node_y + rise + 0.62, top_landing.z),
        normal_axis="y", corner_radius=0.24, corner_segments=5,
        gloss=0.42,
    ))


def _underground_signature(parts, node, tangent, perp, corridor_mid, centres,
                           length, accent, variant):
    """Station-specific architecture for the five underground stops."""
    # Common furniture and a pair of working-scale escalator banks.
    for platform_index, (platform_c, edge_sign) in enumerate(centres):
        escalator = (platform_c + tangent * (length * 0.27 - 9.0)
                     - perp * edge_sign * 0.45)
        _platform_escalator(
            parts, escalator, tangent, perp, node.y, accent,
            -1.0 if platform_index == 0 else 1.0,
        )
        for along in (-length * 0.18, length * 0.15):
            seat = platform_c + tangent * along - perp * edge_sign * 0.65
            # Curved stainless seat shell, legs and end armrests.
            parts.append(mesh.make_rounded_box_x(
                2.8, 0.20, 0.72, (0.30, 0.34, 0.36),
                center=(seat.x, node.y + 1.02, seat.z),
                chamfer=0.09, gloss=0.72,
            ))
            for leg in (-0.92, 0.92):
                lp = seat + tangent * leg
                parts.append(mesh.make_tube_between(
                    (lp.x, node.y + 0.67, lp.z),
                    (lp.x, node.y + 0.96, lp.z),
                    0.045, (0.20, 0.22, 0.23), segments=8, gloss=0.65,
                ))

    if variant == 2:  # Jinghong Road: turquoise wave ceiling.
        color = (0.08, 0.62, 0.66)
        for along in range(int(-length / 2 + 8), int(length / 2 - 7), 10):
            mid = corridor_mid + tangent * float(along)
            wave = 0.18 * math.sin(along * 0.09)
            left = mid - perp * 8.5
            right = mid + perp * 8.5
            crest = mid + glm.vec3(0.0, 6.02 + wave, 0.0)
            parts.append(mesh.make_tube_between(
                (left.x, node.y + 5.70, left.z),
                (crest.x, node.y + 6.02 + wave, crest.z),
                0.085, color, segments=10, gloss=0.68,
            ))
            parts.append(mesh.make_tube_between(
                (crest.x, node.y + 6.02 + wave, crest.z),
                (right.x, node.y + 5.70, right.z),
                0.085, color, segments=10, gloss=0.68,
            ))
        for along in range(int(-length / 2 + 16), int(length / 2 - 15), 28):
            pod = corridor_mid + tangent * float(along)
            parts.append(mesh.make_ellipsoid(
                radii=(2.7, 0.16, 0.72), color=(0.62, 0.88, 0.91),
                center=(pod.x, node.y + 6.12, pod.z), rings=5, segments=14,
                emissive=0.34, gloss=0.70,
            ))

    elif variant == 3:  # Sanlin South: bronze lattice and warm wall wash.
        bronze = (0.56, 0.32, 0.13)
        for side in (-1.0, 1.0):
            wall = corridor_mid + perp * side * 9.30
            for along in range(int(-length / 2 + 8), int(length / 2 - 7), 16):
                a = wall + tangent * float(along - 5)
                b = wall + tangent * float(along + 5)
                parts.append(mesh.make_tube_between(
                    (a.x, node.y + 1.15, a.z), (b.x, node.y + 5.55, b.z),
                    0.055, bronze, segments=9, gloss=0.62,
                ))
                parts.append(mesh.make_tube_between(
                    (a.x, node.y + 5.55, a.z), (b.x, node.y + 1.15, b.z),
                    0.055, bronze, segments=9, gloss=0.62,
                ))

    elif variant == 4:  # Kangqiao East: blue elliptical structural arches.
        arch_color = (0.16, 0.48, 0.78)
        for along in range(int(-length / 2 + 8), int(length / 2 - 7), 14):
            mid = corridor_mid + tangent * float(along)
            left = mid - perp * 8.55
            right = mid + perp * 8.55
            shoulder_l = mid - perp * 3.8
            shoulder_r = mid + perp * 3.8
            parts.append(mesh.make_tube_between(
                (left.x, node.y + 4.55, left.z),
                (shoulder_l.x, node.y + 6.10, shoulder_l.z),
                0.075, arch_color, segments=10, gloss=0.72,
            ))
            parts.append(mesh.make_tube_between(
                (shoulder_l.x, node.y + 6.10, shoulder_l.z),
                (shoulder_r.x, node.y + 6.10, shoulder_r.z),
                0.075, arch_color, segments=10, gloss=0.72,
            ))
            parts.append(mesh.make_tube_between(
                (shoulder_r.x, node.y + 6.10, shoulder_r.z),
                (right.x, node.y + 4.55, right.z),
                0.075, arch_color, segments=10, gloss=0.72,
            ))

    elif variant == 5:  # International Resort: restrained gold/green motif.
        gold = (0.72, 0.50, 0.15)
        for along in range(int(-length / 2 + 10), int(length / 2 - 9), 18):
            mid = corridor_mid + tangent * float(along)
            for side in (-1.0, 1.0):
                root = mid + perp * side * 8.45
                crown = mid + perp * side * 2.1
                parts.append(mesh.make_tube_between(
                    (root.x, node.y + 4.20, root.z),
                    (crown.x, node.y + 6.05, crown.z),
                    0.065, gold, segments=10, gloss=0.72,
                ))
            parts.append(mesh.make_ellipsoid(
                radii=(0.65, 0.12, 0.35), color=(0.28, 0.56, 0.27),
                center=(mid.x, node.y + 6.12, mid.z), rings=4, segments=10,
                emissive=0.10, gloss=0.45,
            ))

    elif variant == 6:  # Pudong airport: silver aero spines and travelators.
        silver = (0.54, 0.60, 0.64)
        for lateral in (-5.9, 0.0, 5.9):
            spine = corridor_mid + perp * lateral
            a = spine - tangent * (length * 0.47)
            b = spine + tangent * (length * 0.47)
            parts.append(mesh.make_tube_between(
                (a.x, node.y + 6.13, a.z), (b.x, node.y + 6.13, b.z),
                0.095, silver, segments=12, gloss=0.78,
            ))
        for platform_c, edge_sign in centres:
            belt = platform_c - perp * edge_sign * 1.15
            parts.append(mesh.make_rounded_box_x(
                length * 0.34, 0.16, 1.25, (0.16, 0.20, 0.22),
                center=(belt.x, node.y + 0.78, belt.z),
                chamfer=0.07, gloss=0.72,
            ))
            for rail_side in (-1.0, 1.0):
                rail = belt + perp * rail_side * 0.72
                a = rail - tangent * (length * 0.17)
                b = rail + tangent * (length * 0.17)
                parts.append(mesh.make_tube_between(
                    (a.x, node.y + 1.58, a.z),
                    (b.x, node.y + 1.58, b.z),
                    0.045, (0.12, 0.14, 0.15), segments=9, gloss=0.76,
                ))


def _station_complex(parts, line, station, variant=0):
    """Full two-platform station with canopies, furniture and glazed hall."""
    if station.underground:
        _underground_station(parts, line, station, variant)
        return
    node, tangent = line.position_at(station.distance)
    tangent = glm.normalize(tangent)
    perp = glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x))
    is_long = ("虹桥" in station.name or "旅游度假区" in station.name
               or "浦东" in station.name)
    length = config.LONG_PLATFORM_LENGTH if is_long else config.STATION_SIZE[0]
    platform_w = 5.6
    platform_off = config.TRAIN_SIZE[2] / 2.0 + 0.18 + platform_w / 2.0
    centres = [
        (node + perp * platform_off, -1.0),
        (node + perp * (config.OPPOSING_TRACK_OFFSET - platform_off), 1.0),
    ]

    for platform_c, edge_sign in centres:
        parts.append(mesh.make_oriented_box(
            size=(length, 0.55, platform_w), color=PLATFORM_COLOR,
            center=(platform_c.x, 0.33, platform_c.z), forward=tangent, gloss=0.16,
        ))
        # Bright tactile warning strip along the track-facing platform edge.
        edge = platform_c + perp * edge_sign * (platform_w / 2.0 - 0.22)
        parts.append(mesh.make_oriented_box(
            size=(length * 0.96, 0.035, 0.34), color=(0.92, 0.73, 0.16),
            center=(edge.x, 0.625, edge.z), forward=tangent, gloss=0.22,
        ))
        # Waist-height platform screen doors as seen at the real airport line.
        screen_edge = platform_c + perp * edge_sign * (platform_w / 2.0 - 0.12)
        for along in range(int(-length / 2 + 3), int(length / 2 - 2), 4):
            panel = screen_edge + tangent * float(along)
            parts.append(mesh.make_oriented_box(
                size=(3.55, 1.35, 0.085), color=(0.25, 0.40, 0.49),
                center=(panel.x, 1.30, panel.z), forward=tangent, gloss=0.97,
                translucency=0.54,
            ))
            for frame_offset in (-1.80, 1.80):
                frame = panel + tangent * frame_offset
                parts.append(mesh.make_cylinder(
                    radius=0.045, length=1.48, axis="y", segments=8,
                    color=(0.045, 0.052, 0.06), center=(frame.x, 1.34, frame.z),
                    gloss=0.55,
                ))
        warning = screen_edge - perp * edge_sign * 0.03
        parts.append(mesh.make_oriented_box(
            size=(length * 0.98, 0.10, 0.11), color=(0.94, 0.74, 0.08),
            center=(warning.x, 1.26, warning.z), forward=tangent,
            emissive=0.12, gloss=0.38,
        ))

        # Glass-and-steel canopy, columns, benches and route signs.
        roof = platform_c + perp * (-edge_sign * 0.35)
        parts.append(mesh.make_rounded_box_x(
            length * 0.64, 0.34, platform_w * 0.76,
            (0.34, 0.42, 0.50), center=(roof.x, 4.35, roof.z),
            chamfer=0.14, gloss=0.86,
        ))
        for along in range(int(-length * 0.29), int(length * 0.30), 20):
            support = platform_c + tangent * along + perp * (-edge_sign * 0.6)
            parts.append(mesh.make_cylinder(
                radius=0.11, length=3.7, axis="y", segments=12,
                color=(0.30, 0.32, 0.34),
                center=(support.x, 2.18, support.z), gloss=0.55,
            ))
        for along in (-length * 0.18, length * 0.16):
            bench = platform_c + tangent * along + perp * (-edge_sign * 0.75)
            parts.append(mesh.make_oriented_box(
                size=(2.3, 0.18, 0.62), color=(0.30, 0.20, 0.13),
                center=(bench.x, 0.92, bench.z), forward=tangent, gloss=0.18,
            ))
            for leg in (-0.8, 0.8):
                lp = bench + tangent * leg
                parts.append(mesh.make_box(
                    size=(0.12, 0.55, 0.12), color=(0.22, 0.23, 0.24),
                    center=(lp.x, 0.69, lp.z), gloss=0.45,
                ))
        for along in (-length * 0.31, length * 0.31):
            sign = platform_c + tangent * along
            parts.append(mesh.make_oriented_box(
                size=(2.8, 0.62, 0.12), color=station.color,
                center=(sign.x, 2.75, sign.z), forward=tangent,
                emissive=0.42, gloss=0.42,
            ))

    # Enclosed ticket hall on the outside of the player platform.
    player_platform = centres[0][0]
    hall = player_platform + perp * (platform_w / 2.0 + 5.2)
    hall_length = min(92.0, length * 0.36)
    hall_h, hall_depth = config.STATION_SIZE[1], config.STATION_SIZE[2]
    parts.append(mesh.make_oriented_box(
        size=(hall_length, hall_h, hall_depth), color=BUILDING_COLOR,
        center=(hall.x, hall_h / 2.0, hall.z), forward=tangent, gloss=0.16,
    ))
    # Curved terminal roof cap and repeated facade mullions reduce the monolith.
    parts.append(mesh.make_rounded_box_x(
        hall_length * 1.02, 1.35, hall_depth * 1.06, (0.48, 0.52, 0.56),
        center=(hall.x, hall_h + 0.55, hall.z), chamfer=0.42, gloss=0.48,
    ))
    # Track-facing curtain wall and two entrance doors.
    glass_c = hall - perp * (hall_depth / 2.0 + 0.05)
    parts.append(mesh.make_oriented_box(
        size=(hall_length * 0.78, hall_h * 0.54, 0.10), color=(0.16, 0.25, 0.34),
        center=(glass_c.x, hall_h * 0.52, glass_c.z), forward=tangent, gloss=0.96,
        translucency=0.40,
    ))
    for along in (-2.2, 2.2):
        door = glass_c + tangent * along
        parts.append(mesh.make_oriented_box(
            size=(1.55, 2.65, 0.13), color=(0.24, 0.38, 0.48),
            center=(door.x, 1.56, door.z), forward=tangent, gloss=0.97,
        ))
    for along in range(int(-hall_length / 2 + 2), int(hall_length / 2 - 1), 3):
        mullion = glass_c + tangent * float(along)
        parts.append(mesh.make_cylinder(
            radius=0.055, length=hall_h * 0.58, axis="y", segments=8,
            color=(0.12, 0.14, 0.16),
            center=(mullion.x, hall_h * 0.52, mullion.z), gloss=0.68,
        ))
    # Suspended bilingual-style wayfinding panels above each platform.
    for platform_c, edge_sign in centres:
        for along in (-length * 0.18, length * 0.18):
            sign = platform_c + tangent * along - perp * edge_sign * 0.55
            parts.append(mesh.make_oriented_box(
                size=(5.8, 0.72, 0.16), color=(0.92, 0.93, 0.92),
                center=(sign.x, 3.45, sign.z), forward=tangent, gloss=0.32,
            ))
            parts.append(mesh.make_oriented_box(
                size=(5.0, 0.13, 0.17), color=station.color,
                center=(sign.x, 3.29, sign.z), forward=tangent,
                emissive=0.45, gloss=0.45,
            ))
    parts.append(mesh.make_oriented_box(
        size=(hall_length + 1.0, 0.55, hall_depth + 0.8), color=station.color,
        center=(hall.x, hall_h + 0.28, hall.z), forward=tangent,
        emissive=0.30, gloss=0.40,
    ))

    # Footbridge makes the opposing platform functionally connected.
    bridge_mid = (centres[0][0] + centres[1][0]) * 0.5 + tangent * (length * 0.36)
    bridge_span = glm.length(centres[0][0] - centres[1][0]) + platform_w * 0.65
    parts.append(mesh.make_oriented_box(
        size=(bridge_span, 0.38, 2.15), color=(0.48, 0.52, 0.56),
        center=(bridge_mid.x, 5.25, bridge_mid.z), forward=perp, gloss=0.34,
    ))
    for centre, _ in centres:
        stair = centre + tangent * (length * 0.36)
        parts.append(mesh.make_box(
            size=(2.0, 4.8, 2.0), color=(0.50, 0.53, 0.55),
            center=(stair.x, 2.4, stair.z), gloss=0.18,
        ))

    if variant == 0:
        # Hongqiao: airport-scale arched steel ribs and a glazed concourse
        # lantern make the terminal station recognisable from a distance.
        for along in range(int(-hall_length / 2 + 3), int(hall_length / 2 - 2), 7):
            rib_mid = hall + tangent * float(along)
            left = rib_mid - perp * hall_depth * 0.52
            right = rib_mid + perp * hall_depth * 0.52
            apex = rib_mid + glm.vec3(0.0, hall_h + 2.35, 0.0)
            parts.append(mesh.make_tube_between(
                (left.x, hall_h - 0.15, left.z), (apex.x, apex.y, apex.z),
                0.075, (0.30, 0.34, 0.37), segments=9, gloss=0.62,
            ))
            parts.append(mesh.make_tube_between(
                (apex.x, apex.y, apex.z), (right.x, hall_h - 0.15, right.z),
                0.075, (0.30, 0.34, 0.37), segments=9, gloss=0.62,
            ))
        lantern = hall + tangent * (hall_length * 0.22)
        parts.append(mesh.make_ellipsoid(
            radii=(7.0, 1.45, 3.4), color=(0.18, 0.34, 0.45),
            center=(lantern.x, hall_h + 1.15, lantern.z),
            rings=7, segments=16, gloss=0.94, translucency=0.36,
        ))
    else:
        # Zhongchun Road: suburban acoustic glazing and a street-facing
        # entrance pylon distinguish it from the airport terminal.
        for side in (-1.0, 1.0):
            barrier_c = (centres[0][0] + centres[1][0]) * 0.5 + perp * side * 10.1
            for along in range(int(-length / 2 + 4), int(length / 2 - 3), 8):
                panel = barrier_c + tangent * float(along)
                parts.append(mesh.make_oriented_box(
                    size=(7.3, 2.25, 0.09), color=(0.20, 0.34, 0.40),
                    center=(panel.x, 1.65, panel.z), forward=tangent,
                    gloss=0.91, translucency=0.30,
                ))
                parts.append(mesh.make_cylinder(
                    radius=0.055, length=2.65, axis="y", segments=8,
                    color=(0.32, 0.35, 0.37),
                    center=(panel.x, 1.52, panel.z), gloss=0.64,
                ))
        pylon = hall + perp * (hall_depth * 0.72)
        parts.append(mesh.make_rounded_box_x(
            3.0, 7.4, 1.2, (0.74, 0.76, 0.77),
            center=(pylon.x, 3.7, pylon.z), chamfer=0.18, gloss=0.34,
        ))
        parts.append(mesh.make_rounded_panel(
            2.15, 2.6, 0.10, station.color,
            center=(pylon.x, 4.35, pylon.z + 0.63), normal_axis="z",
            corner_radius=0.16, emissive=0.34, gloss=0.60,
        ))


def _tunnel(parts, line):
    """Large-bore elliptical tunnel with walkways and continuous light pools."""
    points = line.underground_points()
    if len(points) < 2:
        return
    centre = mesh.offset_path(points, config.OPPOSING_TRACK_OFFSET * 0.5)
    half_width = 8.7
    parts.append(mesh.make_path_ribbon(
        centre, half_width, -0.10, (0.25, 0.27, 0.29), gloss=0.08,
    ))
    parts.append(mesh.make_tunnel_shell(
        centre, horizontal_radius=9.2, vertical_radius=6.7,
        center_height=6.55, color=(0.43, 0.45, 0.47),
        radial_segments=24, gloss=0.17,
    ))
    for lateral in (-7.35, 7.35):
        walkway = mesh.offset_path(centre, lateral)
        parts.append(mesh.make_path_ribbon(
            walkway, 0.75, 0.42, (0.48, 0.49, 0.48), gloss=0.14,
        ))

        # Continuous cable trough with three separately readable conduits.
        tray = mesh.offset_path(centre, lateral - (0.40 if lateral > 0 else -0.40))
        parts.append(mesh.make_extruded_ribbon(
            tray, 0.22, 1.02, 1.15, (0.20, 0.22, 0.23),
            gloss=0.38, closed=False,
        ))
        for cable_offset, cable_color in (
            (-0.12, (0.08, 0.09, 0.10)),
            (0.00, (0.10, 0.10, 0.11)),
            (0.12, (0.55, 0.16, 0.08)),
        ):
            cable = mesh.offset_path(tray, cable_offset)
            parts.append(mesh.make_extruded_ribbon(
                cable, 0.028, 1.16, 1.22, cable_color,
                gloss=0.46, closed=False,
            ))

    # Central drainage channel and cover slabs.
    parts.append(mesh.make_extruded_ribbon(
        centre, 0.34, -0.03, 0.055, (0.12, 0.13, 0.14),
        gloss=0.18, closed=False,
    ))

    total = sum(glm.length(points[i + 1] - points[i]) for i in range(len(points) - 1))
    d = 3.0
    while d < total:
        pos, fwd = _path_pose(points, d)
        perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
        mid = pos + perp * (config.OPPOSING_TRACK_OFFSET * 0.5)
        for lateral in (-5.4, 0.0, 5.4):
            lamp = mid + perp * lateral
            parts.append(mesh.make_oriented_box(
                size=(2.3, 0.10, 0.26), color=(0.88, 0.95, 1.0),
                center=(lamp.x, pos.y + 6.18, lamp.z), forward=fwd,
                emissive=1.0, gloss=0.58,
            ))
        d += 24.0

    # Emergency equipment and cross-passage doors are intentionally less
    # frequent than the light strips, reproducing the long service rhythm of a
    # real inter-airport main-line tunnel.
    d = 120.0
    equipment_index = 0
    while d < total:
        pos, fwd = _path_pose(points, d)
        perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
        mid = pos + perp * (config.OPPOSING_TRACK_OFFSET * 0.5)
        wall_side = 1.0 if equipment_index % 2 == 0 else -1.0
        wall = mid + perp * wall_side * 8.66
        # Red fire cabinet and illuminated evacuation direction plate.
        parts.append(mesh.make_oriented_box(
            size=(0.72, 1.05, 0.18), color=(0.72, 0.055, 0.035),
            center=(wall.x, pos.y + 1.42, wall.z), forward=fwd, gloss=0.58,
        ))
        sign = wall - perp * wall_side * 0.11 + fwd * 0.78
        parts.append(mesh.make_rounded_panel(
            0.68, 0.28, 0.06, (0.10, 0.92, 0.42),
            center=(sign.x, pos.y + 2.48, sign.z), normal_axis="z",
            corner_radius=0.06, emissive=0.92, gloss=0.56,
        ))
        if equipment_index % 4 == 0:
            door = mid + perp * wall_side * 8.58
            parts.append(mesh.make_oriented_box(
                size=(1.45, 2.55, 0.20), color=(0.46, 0.49, 0.50),
                center=(door.x, pos.y + 1.70, door.z), forward=fwd, gloss=0.42,
            ))
            parts.append(mesh.make_rounded_panel(
                0.20, 0.12, 0.05, (0.88, 0.77, 0.16),
                center=(door.x + fwd.x * 0.45, pos.y + 1.65,
                        door.z + fwd.z * 0.45), normal_axis="z",
                corner_radius=0.04, emissive=0.22, gloss=0.66,
            ))
        d += 240.0
        equipment_index += 1

    # Portal frame at the ground-to-tunnel transition.
    portal = centre[0]
    _, portal_fwd = _path_pose(centre, 0.0)
    portal_perp = glm.normalize(glm.vec3(-portal_fwd.z, 0.0, portal_fwd.x))
    parts.append(mesh.make_oriented_box(
        size=(18.2, 0.55, 0.65), color=(0.38, 0.40, 0.42),
        center=(portal.x, portal.y + 6.45, portal.z), forward=portal_perp,
        gloss=0.20,
    ))


def _parked_aircraft(parts, cx, cz, direction=1.0):
    """Compact curved airliner silhouette for airport aprons."""
    fuselage = (0.86, 0.88, 0.90)
    parts.append(mesh.make_ellipsoid(
        radii=(18.0, 2.15, 1.85), color=fuselage,
        center=(cx, 3.35, cz), rings=9, segments=28, gloss=0.88,
    ))
    # Tapered wing planforms use actual quadrilateral surfaces rather than
    # rectangular slabs.
    for side in (-1.0, 1.0):
        root_z = cz + side * 1.05
        tip_z = cz + side * 13.5
        corners = [
            (cx - 5.8, 3.28, root_z), (cx + 4.2, 3.28, root_z),
            (cx + 1.2, 3.20, tip_z), (cx - 1.8, 3.20, tip_z),
        ]
        if side < 0:
            corners.reverse()
        parts.append(mesh.make_quad(
            corners, (0.0, 1.0, 0.0), fuselage, gloss=0.72,
        ))
        # Engine nacelle under each wing.
        parts.append(mesh.make_cylinder(
            radius=0.72, length=2.8, axis="x", segments=18,
            color=(0.26, 0.30, 0.34),
            center=(cx + direction * 0.6, 2.45, cz + side * 6.2), gloss=0.76,
        ))
    # Swept tailplane and vertical stabiliser.
    parts.append(mesh.make_quad(
        [(cx - direction * 15.2, 3.62, cz - 1.0),
         (cx - direction * 11.5, 3.62, cz - 1.0),
         (cx - direction * 13.4, 3.60, cz - 5.0),
         (cx - direction * 15.1, 3.60, cz - 5.0)],
        (0.0, 1.0, 0.0), fuselage, gloss=0.72,
    ))
    parts.append(mesh.make_quad(
        [(cx - direction * 15.0, 3.3, cz),
         (cx - direction * 11.4, 3.3, cz),
         (cx - direction * 14.7, 8.0, cz),
         (cx - direction * 16.2, 7.4, cz)],
        (0.0, 0.0, 1.0), (0.12, 0.37, 0.72), gloss=0.78,
    ))


def _airport_landmark(parts, line, station, pudong=False):
    """Airport terminal, runway, apron, lighting and control tower."""
    node, fwd = line.position_at(station.distance)
    fwd = glm.normalize(glm.vec3(fwd.x, 0.0, fwd.z))
    perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
    side = -1.0 if pudong else 1.0
    runway_mid = glm.vec3(node.x, 0.0, node.z) + perp * side * 460.0
    runway_length = 2350.0 if pudong else 1850.0
    parts.append(mesh.make_oriented_box(
        size=(runway_length, 0.10, 68.0), color=(0.19, 0.20, 0.21),
        center=(runway_mid.x, 0.04, runway_mid.z), forward=fwd, gloss=0.12,
    ))
    # White threshold blocks, centreline and inset edge lights.
    for along in range(int(-runway_length / 2 + 45), int(runway_length / 2 - 44), 85):
        mark = runway_mid + fwd * float(along)
        parts.append(mesh.make_oriented_box(
            size=(10.0, 0.025, 0.58), color=(0.90, 0.91, 0.88),
            center=(mark.x, 0.105, mark.z), forward=fwd, gloss=0.34,
        ))
    for along in range(int(-runway_length / 2), int(runway_length / 2 + 1), 55):
        p = runway_mid + fwd * float(along)
        for edge_side in (-1.0, 1.0):
            lamp = p + perp * edge_side * 32.5
            parts.append(mesh.make_ellipsoid(
                radii=(0.13, 0.10, 0.13), color=(0.74, 0.88, 1.0),
                center=(lamp.x, 0.18, lamp.z), rings=4, segments=8,
                emissive=0.78, gloss=0.82,
            ))

    apron = glm.vec3(node.x, 0.0, node.z) + perp * side * 125.0
    parts.append(mesh.make_oriented_box(
        size=(440.0, 0.075, 155.0), color=(0.40, 0.41, 0.41),
        center=(apron.x, 0.04, apron.z), forward=fwd, gloss=0.10,
    ))
    terminal = glm.vec3(node.x, 0.0, node.z) + perp * side * 52.0
    terminal_len = 300.0 if pudong else 235.0
    parts.append(mesh.make_rounded_panel(
        terminal_len, 44.0, 16.0, (0.61, 0.63, 0.64),
        center=(terminal.x, 8.0, terminal.z), normal_axis="y",
        corner_radius=7.0, corner_segments=7, gloss=0.34,
    ))
    parts.append(mesh.make_rounded_box_x(
        terminal_len * 0.98, 3.8, 43.0, (0.20, 0.32, 0.40),
        center=(terminal.x, 16.2, terminal.z), chamfer=1.8, gloss=0.91,
    ))
    # Curtain-wall bays and boarding piers.
    face = terminal - perp * side * 22.2
    for along in range(int(-terminal_len / 2 + 6), int(terminal_len / 2 - 5), 12):
        bay = face + fwd * float(along)
        parts.append(mesh.make_rounded_panel(
            9.5, 8.2, 0.10, (0.10, 0.23, 0.31),
            center=(bay.x, 7.2, bay.z), normal_axis="z",
            corner_radius=0.42, corner_segments=5, gloss=0.96,
            translucency=0.40,
        ))
    tower = terminal - fwd * (terminal_len * 0.38) + perp * side * 58.0
    parts.append(mesh.make_cylinder(
        radius=3.1, length=31.0, axis="y", segments=22,
        color=(0.54, 0.55, 0.56), center=(tower.x, 15.5, tower.z), gloss=0.26,
    ))
    parts.append(mesh.make_ellipsoid(
        radii=(6.2, 3.1, 6.2), color=(0.08, 0.18, 0.24),
        center=(tower.x, 31.5, tower.z), rings=7, segments=20,
        gloss=0.94, translucency=0.12,
    ))
    for aircraft_offset in (-105.0, 75.0):
        aircraft = apron + fwd * aircraft_offset + perp * side * 18.0
        _parked_aircraft(parts, aircraft.x, aircraft.z, direction=1.0)


def _airport_scenery(parts, line):
    _airport_landmark(parts, line, line.stations[0], pudong=False)
    _airport_landmark(parts, line, line.stations[-1], pudong=True)


def build_static_mesh(line, stations) -> Mesh:
    parts = []
    rng = random.Random(SCENERY_SEED)

    # Ground plane, centred on the line, large enough to contain it.
    c = line.centroid()
    span = 0.0
    for p in line.points:
        span = max(span, glm.length(p - c))
    ground_size = max(300.0, span * 2.5 + 120.0)
    parts.append(
        mesh.make_ground(ground_size, center=(c.x, c.z), color=config.GROUND_COLOR, y=0.0)
    )

    # Grass verge plus a complete 3D permanent-way assembly.
    # Surface verge follows the alignment; tunnel geometry hides it underground.
    _surface_verges(parts, line)
    _main_track(parts, line)
    _tunnel(parts, line)

    # Relief line and depot sidings.  Wide ballast throats make the junctions
    # read as real turnouts rather than disconnected decorative rails.
    for branch in config.BRANCH_LINES:
        pts = [glm.vec3(x, 0.0, z) for x, z in branch]
        _branch_track(parts, pts)
        _turnout_detail(parts, pts)
    _route_crossovers(parts, line)

    # Electrification and line-side greenery (like the reference scenery).
    _catenary(parts, line)
    _signal_housings(parts, line)
    _lineside_trees(parts, line, rng)
    _airport_scenery(parts, line)

    for station_index, station in enumerate(stations):
        _station_complex(parts, line, station, station_index)

    # Distant rolling terrain frames the larger map without blocking the line.
    for hx, hz, rx, ry, rz in [
        (7200, 620, 520, 72, 360), (12800, -540, 440, 58, 310),
        (22600, 880, 610, 82, 420), (31800, -720, 540, 66, 390),
        (47200, 960, 720, 94, 480), (55200, -680, 560, 76, 410),
    ]:
        parts.append(mesh.make_ellipsoid(
            radii=(rx, ry, rz), color=(0.23, 0.34, 0.16),
            center=(hx, -ry * 0.72, hz), rings=7, segments=14,
        ))

    # A reflective river basin and rocky embankments give the expanded map
    # distinct terrain zones instead of a single flat green plane.
    parts.append(mesh.make_ground(
        1250.0, center=(30200.0, -1550.0), color=(0.16, 0.32, 0.43), y=0.045, gloss=0.88,
    ))
    for rx, rz, sx, sy, sz in [
        (29600, -920, 260, 46, 180), (30700, -880, 230, 40, 165),
        (42100, 1120, 320, 54, 210), (42900, 1260, 280, 48, 190),
    ]:
        parts.append(mesh.make_ellipsoid(
            radii=(sx, sy, sz), color=(0.31, 0.30, 0.27),
            center=(rx, -sy * 0.35, rz), rings=6, segments=12, gloss=0.08,
        ))

    return mesh.combine(parts)


def _build_legacy_train_mesh() -> Mesh:
    """Detailed modern metro car; instanced into a three-car consist."""
    length, height, width = config.TRAIN_SIZE
    body_length = length - 5.0
    nose_length = 2.5
    base_y = height / 2 + 0.68
    parts = [
        # Chamfered aluminium body: roof/lower corners now catch separate light.
        mesh.make_rounded_box_x(
            length=body_length, height=height, width=width,
            color=config.TRAIN_COLOR, center=(0.0, base_y, 0.0),
            chamfer=0.34, gloss=0.86,
        ),
        # Lower skirt and under-frame equipment.
        mesh.make_box(
            size=(length * 0.94, 0.48, width * 0.90), color=(0.30, 0.32, 0.34),
            center=(0.0, 0.88, 0.0), gloss=0.46,
        ),
        mesh.make_box(
            size=(2.2, 0.55, width * 0.74), color=(0.13, 0.15, 0.17),
            center=(-1.8, 0.76, 0.0), gloss=0.28,
        ),
        mesh.make_box(
            size=(1.8, 0.48, width * 0.72), color=(0.17, 0.19, 0.21),
            center=(1.2, 0.76, 0.0), gloss=0.30,
        ),
        # Airport Link Line blue route stripe is inset into the bodyside.
        mesh.make_box(
            size=(body_length * 1.003, 0.30, width * 1.004), color=(0.035, 0.34, 0.76),
            center=(0.0, base_y - height * 0.20, 0.0), gloss=0.80,
        ),
        mesh.make_box(
            size=(body_length * 1.004, 0.085, width * 1.005), color=(0.78, 0.52, 0.14),
            center=(0.0, base_y - height * 0.285, 0.0), gloss=0.72,
        ),
        # Rounded-looking roof cap and HVAC packs.
        mesh.make_rounded_box_x(
            body_length * 0.96, 0.48, width * 0.86, (0.43, 0.46, 0.49),
            center=(0.0, base_y + height / 2 + 0.08, 0.0), chamfer=0.16, gloss=0.30,
        ),
    ]

    # Tapered, chamfered cab noses replace the old flat box ends.
    for end in (-1.0, 1.0):
        nose_centre = end * (body_length / 2.0 + nose_length / 2.0)
        parts.append(mesh.make_tapered_rounded_x(
            nose_length, height * 0.96, width * 0.98, config.TRAIN_COLOR,
            center=(nose_centre, base_y - 0.03, 0.0), narrow_end=int(end),
            nose_scale=0.70, chamfer=0.32, gloss=0.88,
        ))

    # Individually modelled windows and passenger doors on both sides.
    door_positions = (-7.20, -2.40, 2.40, 7.20)
    window_positions = (-9.45, -5.00, 0.0, 5.00, 9.45)
    side_z = width / 2 + 0.035
    for side in (-1.0, 1.0):
        z = side * side_z
        for x in window_positions:
            parts.append(mesh.make_box(
                size=(1.12, 0.88, 0.075), color=(0.075, 0.13, 0.18),
                center=(x, base_y + 0.48, z), gloss=0.98,
            ))
        for x in door_positions:
            parts.append(mesh.make_box(
                size=(1.34, 2.35, 0.085), color=(0.62, 0.65, 0.67),
                center=(x, base_y - 0.13, z), gloss=0.78,
            ))
            parts.append(mesh.make_box(
                size=(0.92, 0.82, 0.095), color=(0.07, 0.13, 0.17),
                center=(x, base_y + 0.42, z + side * 0.008), gloss=0.98,
            ))
            # Door seam and warning strip.
            parts.append(mesh.make_box(
                size=(0.035, 2.18, 0.102), color=(0.16, 0.18, 0.19),
                center=(x, base_y - 0.13, z + side * 0.012), gloss=0.50,
            ))

    # Cab windscreens, destination displays, headlights and tail lamps.
    for end, lamp_color in ((1.0, (1.0, 0.98, 0.86)), (-1.0, (0.95, 0.08, 0.04))):
        x = end * (length / 2 + 0.035)
        parts.append(mesh.make_box(
            size=(0.105, 1.15, width * 0.80), color=(0.025, 0.29, 0.70),
            center=(x, base_y - 0.28, 0.0), gloss=0.88,
        ))
        parts.append(mesh.make_box(
            size=(0.112, 0.09, width * 0.72), color=(0.80, 0.52, 0.14),
            center=(x + end * 0.008, base_y - 0.83, 0.0), gloss=0.75,
        ))
        parts.append(mesh.make_box(
            size=(0.09, 1.06, width * 0.66), color=(0.045, 0.08, 0.11),
            center=(x, base_y + 0.50, 0.0), gloss=0.99,
        ))
        parts.append(mesh.make_box(
            size=(0.10, 0.31, width * 0.48), color=(1.0, 0.62, 0.08),
            center=(x + end * 0.01, base_y + 1.30, 0.0), emissive=0.82, gloss=0.5,
        ))
        for zoff in (-width * 0.34, width * 0.34):
            parts.append(mesh.make_box(
                size=(0.11, 0.25, 0.32), color=lamp_color,
                center=(x + end * 0.02, base_y - 0.85, zoff),
                emissive=1.0, gloss=0.86,
            ))

    # Bogies, wheel faces and suspension detail.
    for xoff in (-length * 0.29, length * 0.29):
        parts.append(mesh.make_box(
            size=(2.15, 0.38, width * 0.73), color=(0.085, 0.095, 0.105),
            center=(xoff, 0.72, 0.0), gloss=0.40,
        ))
        for wheel_x in (-0.82, 0.82):
            parts.append(mesh.make_cylinder(
                radius=0.12, length=width * 0.84, axis="z",
                color=(0.10, 0.11, 0.12), center=(xoff + wheel_x, 0.57, 0.0),
                segments=12, gloss=0.55,
            ))
            for zoff in (-width * 0.44, width * 0.44):
                parts.append(mesh.make_cylinder(
                    radius=0.43, length=0.20, axis="z", segments=18,
                    color=(0.10, 0.11, 0.12),
                    center=(xoff + wheel_x, 0.57, zoff), gloss=0.78,
                ))
                parts.append(mesh.make_cylinder(
                    radius=0.22, length=0.215, axis="z", segments=14,
                    color=(0.32, 0.33, 0.34),
                    center=(xoff + wheel_x, 0.57, zoff), gloss=0.70,
                ))

    # Roof equipment and a simple pantograph silhouette.
    roof_y = base_y + height / 2 + 0.44
    for x in (-2.8, 1.4):
        parts.append(mesh.make_box(
            size=(1.65, 0.30, width * 0.55), color=(0.31, 0.34, 0.36),
            center=(x, roof_y, 0.0), gloss=0.28,
        ))
    for x in (-0.55, 0.55):
        parts.append(mesh.make_box(
            size=(0.10, 0.95, 0.10), color=(0.18, 0.19, 0.20),
            center=(x, roof_y + 0.55, 0.0), gloss=0.60,
        ))
    parts.append(mesh.make_box(
        size=(1.65, 0.09, 0.12), color=(0.12, 0.13, 0.14),
        center=(0.0, roof_y + 1.05, 0.0), gloss=0.75,
    ))

    # Couplers make the gaps between independently aligned cars believable.
    for end in (-1.0, 1.0):
        parts.append(mesh.make_box(
            size=(0.62, 0.22, 0.34), color=(0.08, 0.09, 0.10),
            center=(end * (length / 2 + 0.28), 0.68, 0.0), gloss=0.52,
        ))
    return mesh.combine(parts)


def build_train_mesh(car_type="middle") -> Mesh:
    """High-detail Airport Link Line car.

    ``lead`` has its aerodynamic cab at local +X, ``tail`` at local -X and
    ``middle`` has gangways at both ends.  Keeping these as separate GPU meshes
    prevents every vehicle in the four-car formation from looking like an
    independently coupled cab car.
    """
    if car_type not in {"lead", "middle", "tail"}:
        raise ValueError(f"Unknown train car type: {car_type}")

    length, height, width = config.TRAIN_SIZE
    cab_sign = 1.0 if car_type == "lead" else (-1.0 if car_type == "tail" else 0.0)
    nose_length = 3.35
    if cab_sign:
        body_length = length - nose_length
        body_centre_x = -cab_sign * nose_length * 0.5
    else:
        body_length = length - 0.30
        body_centre_x = 0.0
    base_y = height * 0.5 + 0.68
    white = config.TRAIN_COLOR
    blue = (0.025, 0.31, 0.76)
    gold = (0.82, 0.54, 0.13)
    glass = (0.025, 0.075, 0.105)
    dark_metal = (0.075, 0.085, 0.095)

    parts = [
        mesh.make_superellipse_body_x(
            body_length, height, width, white,
            center=(body_centre_x, base_y, 0.0), segments=28,
            exponent=4.2, gloss=0.90,
        ),
        mesh.make_superellipse_body_x(
            body_length * 0.97, 0.48, width * 0.88, (0.42, 0.45, 0.48),
            center=(body_centre_x, base_y + height * 0.52, 0.0),
            segments=20, exponent=3.2, gloss=0.34,
        ),
        mesh.make_rounded_box_x(
            body_length * 0.96, 0.46, width * 0.90, (0.22, 0.24, 0.26),
            center=(body_centre_x, 0.88, 0.0), chamfer=0.16, gloss=0.48,
        ),
    ]

    if cab_sign:
        base_x = body_centre_x + cab_sign * body_length * 0.5
        parts.append(mesh.make_streamlined_nose_x(
            base_x=base_x, end_sign=int(cab_sign), length=nose_length,
            height=height, width=width, color=white, base_y=base_y,
            segments=28, rings=8, exponent=4.0, gloss=0.92,
        ))

    # Properly inset bodyside livery and glazing, all with radiused corners.
    side_z = width * 0.5 + 0.024
    for side in (-1.0, 1.0):
        z = side * side_z
        parts.append(mesh.make_rounded_panel(
            body_length * 0.985, 0.29, 0.045, blue,
            center=(body_centre_x, base_y - height * 0.20, z),
            normal_axis="z", corner_radius=0.09, corner_segments=4, gloss=0.84,
        ))
        parts.append(mesh.make_rounded_panel(
            body_length * 0.982, 0.075, 0.048, gold,
            center=(body_centre_x, base_y - height * 0.295, z + side * 0.003),
            normal_axis="z", corner_radius=0.03, corner_segments=3, gloss=0.78,
        ))

        door_positions = [body_centre_x + q for q in (-7.15, -2.40, 2.40, 7.15)]
        door_positions = [x for x in door_positions
                          if abs(x - body_centre_x) < body_length * 0.5 - 0.85]
        window_positions = [body_centre_x + q for q in (-9.35, -5.0, 0.0, 5.0, 9.35)]
        window_positions = [x for x in window_positions
                            if abs(x - body_centre_x) < body_length * 0.5 - 0.52]
        for x in window_positions:
            parts.append(mesh.make_rounded_panel(
                1.18, 0.88, 0.065, glass,
                center=(x, base_y + 0.49, z), normal_axis="z",
                corner_radius=0.17, corner_segments=5, gloss=0.99,
                translucency=0.18,
            ))
        for x in door_positions:
            parts.append(mesh.make_rounded_panel(
                1.40, 2.36, 0.072, (0.63, 0.66, 0.68),
                center=(x, base_y - 0.12, z), normal_axis="z",
                corner_radius=0.10, corner_segments=4, gloss=0.82,
            ))
            parts.append(mesh.make_rounded_panel(
                0.96, 0.84, 0.082, glass,
                center=(x, base_y + 0.43, z + side * 0.008), normal_axis="z",
                corner_radius=0.15, corner_segments=5, gloss=0.99,
                translucency=0.16,
            ))
            parts.append(mesh.make_rounded_panel(
                0.038, 2.13, 0.085, (0.13, 0.14, 0.15),
                center=(x, base_y - 0.13, z + side * 0.012), normal_axis="z",
                corner_radius=0.015, corner_segments=2, gloss=0.56,
            ))
            # Recessed push button and amber door-status lamp.
            parts.append(mesh.make_ellipsoid(
                radii=(0.055, 0.055, 0.028), color=(0.10, 0.12, 0.13),
                center=(x + 0.54, base_y - 0.18, z + side * 0.055),
                rings=4, segments=10, gloss=0.82,
            ))
            parts.append(mesh.make_ellipsoid(
                radii=(0.050, 0.030, 0.026), color=(1.0, 0.56, 0.05),
                center=(x, base_y + 1.28, z + side * 0.056),
                rings=4, segments=10, gloss=0.72, emissive=0.75,
            ))

    # Sculpted cab mask, windshield, display, lamps and twin wipers.
    if cab_sign:
        x = cab_sign * (length * 0.5 + 0.018)
        parts.append(mesh.make_rounded_panel(
            width * 0.78, 1.72, 0.075, blue,
            center=(x, base_y + 0.06, 0.0), normal_axis="x",
            corner_radius=0.34, corner_segments=6, gloss=0.90,
        ))
        parts.append(mesh.make_rounded_panel(
            width * 0.66, 1.04, 0.086, glass,
            center=(x + cab_sign * 0.012, base_y + 0.54, 0.0), normal_axis="x",
            corner_radius=0.25, corner_segments=7, gloss=0.995,
            translucency=0.12,
        ))
        parts.append(mesh.make_rounded_panel(
            width * 0.47, 0.28, 0.090, (1.0, 0.56, 0.06),
            center=(x + cab_sign * 0.018, base_y + 1.31, 0.0), normal_axis="x",
            corner_radius=0.08, corner_segments=4, gloss=0.62, emissive=0.88,
        ))
        parts.append(mesh.make_rounded_panel(
            width * 0.72, 0.075, 0.082, gold,
            center=(x + cab_sign * 0.019, base_y - 0.78, 0.0), normal_axis="x",
            corner_radius=0.03, corner_segments=3, gloss=0.78,
        ))
        for zoff in (-width * 0.30, width * 0.30):
            parts.append(mesh.make_ellipsoid(
                radii=(0.075, 0.17, 0.20), color=(1.0, 0.97, 0.84),
                center=(x + cab_sign * 0.055, base_y - 0.72, zoff),
                rings=6, segments=14, gloss=0.92, emissive=1.0,
            ))
            parts.append(mesh.make_ellipsoid(
                radii=(0.070, 0.075, 0.105), color=(0.92, 0.045, 0.025),
                center=(x + cab_sign * 0.052, base_y - 0.38, zoff * 1.08),
                rings=5, segments=12, gloss=0.86, emissive=0.58,
            ))
        for zoff in (-0.48, 0.48):
            parts.append(mesh.make_tube_between(
                (x + cab_sign * 0.07, base_y + 0.28, zoff),
                (x + cab_sign * 0.075, base_y + 0.72, zoff * 0.42),
                0.022, (0.055, 0.06, 0.065), segments=8, gloss=0.70,
            ))

    # Flat inter-car ends receive flexible gangway bellows, cables and coupler.
    gangway_ends = [-1.0, 1.0] if not cab_sign else [-cab_sign]
    for end in gangway_ends:
        end_x = end * (length * 0.5 - 0.03)
        for ring in range(4):
            rx = end_x + end * ring * 0.075
            for zoff in (-1.08, 1.08):
                parts.append(mesh.make_tube_between(
                    (rx, 1.05, zoff), (rx, 3.35, zoff), 0.035,
                    (0.12, 0.13, 0.14), segments=8, gloss=0.36,
                ))
            for yoff in (1.05, 3.35):
                parts.append(mesh.make_tube_between(
                    (rx, yoff, -1.08), (rx, yoff, 1.08), 0.035,
                    (0.12, 0.13, 0.14), segments=8, gloss=0.36,
                ))
        parts.append(mesh.make_cylinder(
            radius=0.15, length=0.64, axis="x", segments=14,
            color=dark_metal, center=(end * (length * 0.5 + 0.28), 0.69, 0.0),
            gloss=0.58,
        ))
        for side in (-1.0, 1.0):
            parts.append(mesh.make_tube_between(
                (end_x, 0.86, side * 0.26),
                (end * (length * 0.5 + 0.43), 0.58, side * 0.34),
                0.035, (0.09, 0.10, 0.11), segments=8, gloss=0.42,
            ))

    # Fully readable bogies: frame, axles, wheel tyres, brake discs, axleboxes
    # and suspension springs instead of four cylinders under a box.
    for bogie_x in (-length * 0.29, length * 0.29):
        parts.append(mesh.make_rounded_box_x(
            2.42, 0.34, width * 0.68, dark_metal,
            center=(bogie_x, 0.78, 0.0), chamfer=0.11, gloss=0.42,
        ))
        for side in (-1.0, 1.0):
            zframe = side * (config.RAIL_HALF_WIDTH + 0.18)
            parts.append(mesh.make_tube_between(
                (bogie_x - 1.02, 0.76, zframe), (bogie_x + 1.02, 0.76, zframe),
                0.10, (0.11, 0.12, 0.13), segments=10, gloss=0.48,
            ))
            for sx in (-0.40, 0.40):
                parts.append(mesh.make_cylinder(
                    radius=0.11, length=0.34, axis="y", segments=12,
                    color=(0.18, 0.19, 0.20),
                    center=(bogie_x + sx, 0.91, zframe), gloss=0.58,
                ))
        for wheel_dx in (-0.86, 0.86):
            axle_x = bogie_x + wheel_dx
            parts.append(mesh.make_cylinder(
                radius=0.105, length=config.RAIL_HALF_WIDTH * 2.0 + 0.34,
                axis="z", segments=12,
                color=(0.16, 0.17, 0.18), center=(axle_x, 0.57, 0.0), gloss=0.56,
            ))
            for side in (-1.0, 1.0):
                wheel_z = side * config.RAIL_HALF_WIDTH
                parts.append(mesh.make_cylinder(
                    radius=0.43, length=0.19, axis="z", segments=22,
                    color=(0.075, 0.080, 0.085),
                    center=(axle_x, 0.57, wheel_z), gloss=0.80,
                ))
                parts.append(mesh.make_cylinder(
                    radius=0.29, length=0.205, axis="z", segments=20,
                    color=(0.44, 0.43, 0.40),
                    center=(axle_x, 0.57, wheel_z), gloss=0.72,
                ))
                parts.append(mesh.make_cylinder(
                    radius=0.14, length=0.225, axis="z", segments=14,
                    color=(0.14, 0.15, 0.16),
                    center=(axle_x, 0.57, wheel_z), gloss=0.58,
                ))

    # Underfloor traction equipment with curved corners and visible conduits.
    for xoff, box_len, tint in (
        (-4.5, 2.7, (0.16, 0.18, 0.19)),
        (-1.2, 2.1, (0.12, 0.14, 0.15)),
        (2.1, 2.5, (0.18, 0.19, 0.20)),
        (5.1, 1.9, (0.13, 0.15, 0.16)),
    ):
        parts.append(mesh.make_rounded_box_x(
            box_len, 0.48, width * 0.68, tint,
            center=(xoff, 0.82, 0.0), chamfer=0.10, gloss=0.34,
        ))
    parts.append(mesh.make_tube_between(
        (-7.8, 0.68, -1.08), (7.8, 0.68, -1.08), 0.045,
        (0.62, 0.19, 0.06), segments=9, gloss=0.48,
    ))

    # Roof HVAC fan housings.  Intermediate cars carry a true diamond-frame
    # pantograph with ceramic insulators and a wide carbon contact strip.
    roof_y = base_y + height * 0.5 + 0.35
    for xoff in (body_centre_x - 4.2, body_centre_x + 4.0):
        parts.append(mesh.make_rounded_box_x(
            2.65, 0.30, width * 0.62, (0.32, 0.35, 0.37),
            center=(xoff, roof_y, 0.0), chamfer=0.12, gloss=0.30,
        ))
        for fan_x in (-0.58, 0.58):
            parts.append(mesh.make_cylinder(
                radius=0.36, length=0.06, axis="y", segments=18,
                color=(0.17, 0.19, 0.20),
                center=(xoff + fan_x, roof_y + 0.18, 0.0), gloss=0.46,
            ))
    if car_type == "middle":
        panto_x = 0.0
        for dx in (-0.62, 0.62):
            parts.append(mesh.make_cylinder(
                radius=0.12, length=0.28, axis="y", segments=14,
                color=(0.66, 0.57, 0.42),
                center=(panto_x + dx, roof_y + 0.32, 0.0), gloss=0.58,
            ))
        lower_y = roof_y + 0.45
        top_y = WIRE_HEIGHT - 0.035
        elbow_y = (lower_y + top_y) * 0.5
        arms = [
            ((panto_x - 0.68, lower_y, 0.0), (panto_x + 0.25, elbow_y, 0.0)),
            ((panto_x + 0.68, lower_y, 0.0), (panto_x - 0.25, elbow_y, 0.0)),
            ((panto_x + 0.25, elbow_y, 0.0), (panto_x - 0.62, top_y, 0.0)),
            ((panto_x - 0.25, elbow_y, 0.0), (panto_x + 0.62, top_y, 0.0)),
        ]
        for start, end in arms:
            parts.append(mesh.make_tube_between(
                start, end, 0.045, (0.17, 0.18, 0.19), segments=10, gloss=0.72,
            ))
        parts.append(mesh.make_tube_between(
            (panto_x - 1.02, top_y, 0.0), (panto_x + 1.02, top_y, 0.0),
            0.055, (0.075, 0.08, 0.085), segments=10, gloss=0.78,
        ))

    return mesh.combine(parts)


def build_cab_interior_mesh() -> Mesh:
    """Driver's cab rendered only in first-person DRIVE mode.

    Coordinates are local to the leading car (+X points through the windscreen).
    The geometry deliberately leaves the windscreen and side-window apertures
    open so the world remains crisp while the structural frame, desk and
    equipment provide a convincing enclosed driving position.
    """
    length, _, width = config.TRAIN_SIZE
    cab_front = length * 0.5 - 0.30
    cab_rear = 6.65
    cab_mid = (cab_front + cab_rear) * 0.5
    cab_length = cab_front - cab_rear
    dark = (0.055, 0.065, 0.075)
    trim = (0.23, 0.26, 0.28)
    metal = (0.40, 0.43, 0.45)
    desk = (0.15, 0.18, 0.20)
    blue = (0.035, 0.30, 0.72)
    parts = [
        # Anti-slip floor, acoustic ceiling and rear bulkhead.
        mesh.make_rounded_box_x(
            cab_length, 0.18, width * 0.91, (0.10, 0.11, 0.12),
            center=(cab_mid, 0.98, 0.0), chamfer=0.10, gloss=0.16,
        ),
        mesh.make_rounded_box_x(
            cab_length, 0.20, width * 0.91, (0.64, 0.66, 0.67),
            center=(cab_mid, 4.34, 0.0), chamfer=0.10, gloss=0.26,
        ),
        mesh.make_rounded_panel(
            width * 0.90, 3.18, 0.16, (0.48, 0.50, 0.51),
            center=(cab_rear, 2.62, 0.0), normal_axis="x",
            corner_radius=0.22, corner_segments=6, gloss=0.24,
        ),
        # Lower side panels below the window line.
        mesh.make_rounded_box_x(
            cab_length * 0.93, 1.12, 0.16, (0.34, 0.36, 0.37),
            center=(cab_mid, 1.57, width * 0.46), chamfer=0.07, gloss=0.30,
        ),
        mesh.make_rounded_box_x(
            cab_length * 0.93, 1.12, 0.16, (0.34, 0.36, 0.37),
            center=(cab_mid, 1.57, -width * 0.46), chamfer=0.07, gloss=0.30,
        ),
    ]

    # Rear access door, glazed insert, latch and warning strip.
    parts.append(mesh.make_rounded_panel(
        1.22, 2.62, 0.08, (0.30, 0.33, 0.35),
        center=(cab_rear - 0.09, 2.48, 0.0), normal_axis="x",
        corner_radius=0.13, corner_segments=5, gloss=0.42,
    ))
    parts.append(mesh.make_rounded_panel(
        0.76, 0.92, 0.09, (0.055, 0.12, 0.16),
        center=(cab_rear - 0.14, 3.10, 0.0), normal_axis="x",
        corner_radius=0.15, corner_segments=6, gloss=0.98,
    ))
    parts.append(mesh.make_cylinder(
        radius=0.055, length=0.10, axis="x", segments=12,
        color=(0.68, 0.69, 0.68),
        center=(cab_rear - 0.17, 2.35, -0.44), gloss=0.80,
    ))

    # Windscreen structure: swept A-pillars, header, sill and central mullion.
    front_low_x = cab_front - 0.28
    front_high_x = cab_front - 0.78
    for side in (-1.0, 1.0):
        parts.append(mesh.make_tube_between(
            (front_low_x, 2.12, side * 1.43),
            (front_high_x, 4.18, side * 1.20),
            0.095, dark, segments=12, gloss=0.50,
        ))
    parts.append(mesh.make_tube_between(
        (front_high_x, 4.18, -1.20), (front_high_x, 4.18, 1.20),
        0.090, dark, segments=12, gloss=0.50,
    ))
    parts.append(mesh.make_tube_between(
        (front_low_x, 2.12, -1.43), (front_low_x, 2.12, 1.43),
        0.085, dark, segments=12, gloss=0.50,
    ))
    parts.append(mesh.make_tube_between(
        (front_low_x - 0.08, 2.20, 0.0),
        (front_high_x - 0.04, 4.12, 0.0),
        0.036, (0.09, 0.10, 0.11), segments=10, gloss=0.58,
    ))

    # Side-window frames and grab rails remain visible during free mouse-look.
    for side in (-1.0, 1.0):
        z = side * width * 0.475
        for y in (2.18, 4.10):
            parts.append(mesh.make_tube_between(
                (cab_rear + 0.34, y, z), (front_high_x - 0.08, y, z),
                0.052, dark, segments=10, gloss=0.54,
            ))
        parts.append(mesh.make_tube_between(
            (cab_rear + 0.34, 2.18, z), (cab_rear + 0.34, 4.10, z),
            0.060, dark, segments=10, gloss=0.54,
        ))
        parts.append(mesh.make_tube_between(
            (front_high_x - 0.08, 2.18, z), (front_high_x - 0.08, 4.10, z),
            0.060, dark, segments=10, gloss=0.54,
        ))
        hand_z = z - side * 0.10
        parts.append(mesh.make_tube_between(
            (cab_rear + 0.72, 1.30, hand_z),
            (cab_rear + 0.72, 2.20, hand_z),
            0.035, metal, segments=9, gloss=0.76,
        ))

    # Sculpted driver's desk and raised instrument binnacle.
    parts.append(mesh.make_rounded_box_x(
        2.85, 0.70, width * 0.86, desk,
        center=(9.95, 1.52, 0.0), chamfer=0.20, gloss=0.32,
    ))
    parts.append(mesh.make_rounded_box_x(
        1.52, 0.32, width * 0.79, (0.21, 0.24, 0.26),
        center=(9.28, 2.04, 0.0), chamfer=0.12, gloss=0.40,
    ))
    parts.append(mesh.make_rounded_panel(
        2.92, 1.18, 0.16, (0.105, 0.125, 0.14),
        center=(9.13, 2.50, 0.0), normal_axis="x",
        corner_radius=0.24, corner_segments=7, gloss=0.36,
    ))

    # Three independent ATO/ATP/TCMS displays with bezels and status bars.
    screen_specs = [
        (-0.91, 0.82, 0.63, (0.08, 0.62, 0.88)),
        (0.00, 0.98, 0.72, (0.12, 0.86, 0.48)),
        (0.98, 0.78, 0.60, (1.00, 0.56, 0.08)),
    ]
    for zoff, screen_w, screen_h, glow in screen_specs:
        parts.append(mesh.make_rounded_panel(
            screen_w + 0.12, screen_h + 0.12, 0.075, (0.025, 0.03, 0.035),
            center=(9.03, 2.55, zoff), normal_axis="x",
            corner_radius=0.11, corner_segments=5, gloss=0.72,
        ))
        parts.append(mesh.make_rounded_panel(
            screen_w, screen_h, 0.08, (glow[0] * 0.18, glow[1] * 0.18, glow[2] * 0.18),
            center=(8.985, 2.55, zoff), normal_axis="x",
            corner_radius=0.09, corner_segments=5, gloss=0.96,
            emissive=0.52,
        ))
        # Data bands make each display read as an active interface.
        for row in (-0.16, 0.00, 0.16):
            parts.append(mesh.make_rounded_panel(
                screen_w * (0.58 + 0.12 * (row + 0.16)), 0.035, 0.085, glow,
                center=(8.94, 2.55 + row, zoff), normal_axis="x",
                corner_radius=0.015, corner_segments=2, gloss=0.70,
                emissive=0.86,
            ))

    # Illuminated button bank and guarded emergency controls.
    button_colors = (
        (0.14, 0.78, 0.42), (0.12, 0.52, 0.94), (0.98, 0.68, 0.08),
        (0.82, 0.12, 0.08), (0.14, 0.78, 0.42), (0.12, 0.52, 0.94),
    )
    for index, color in enumerate(button_colors):
        z = -1.02 + index * 0.40
        parts.append(mesh.make_cylinder(
            radius=0.060, length=0.065, axis="x", segments=12, color=color,
            center=(8.92, 1.91, z), emissive=0.58, gloss=0.82,
        ))
    parts.append(mesh.make_cylinder(
        radius=0.115, length=0.08, axis="x", segments=16,
        color=(0.88, 0.055, 0.035), center=(8.90, 1.87, 1.28),
        emissive=0.20, gloss=0.72,
    ))

    # Separate traction/brake handles, reverser and horn mushroom.
    for zoff, handle_color in ((-0.82, blue), (0.82, (0.72, 0.22, 0.10))):
        parts.append(mesh.make_cylinder(
            radius=0.13, length=0.08, axis="y", segments=14,
            color=(0.08, 0.09, 0.10), center=(9.55, 2.13, zoff), gloss=0.58,
        ))
        parts.append(mesh.make_tube_between(
            (9.55, 2.17, zoff), (9.34, 2.62, zoff),
            0.045, metal, segments=10, gloss=0.76,
        ))
        parts.append(mesh.make_ellipsoid(
            radii=(0.15, 0.11, 0.13), color=handle_color,
            center=(9.31, 2.68, zoff), rings=5, segments=12, gloss=0.76,
        ))
    parts.append(mesh.make_tube_between(
        (9.78, 2.12, 0.28), (9.67, 2.43, 0.28),
        0.035, metal, segments=9, gloss=0.72,
    ))

    # Overhead electrical cabinet, cab light and diagnostic tell-tales.
    parts.append(mesh.make_rounded_box_x(
        1.85, 0.22, 1.55, (0.28, 0.31, 0.33),
        center=(8.05, 4.15, 0.0), chamfer=0.08, gloss=0.34,
    ))
    parts.append(mesh.make_rounded_panel(
        1.16, 0.22, 0.07, (0.88, 0.94, 1.0),
        center=(8.25, 4.025, 0.0), normal_axis="y",
        corner_radius=0.07, corner_segments=4, emissive=0.82, gloss=0.58,
    ))
    for zoff, color in ((-0.52, (0.12, 0.88, 0.40)),
                        (-0.26, (1.0, 0.62, 0.08)),
                        (0.0, (0.12, 0.56, 0.96)),
                        (0.26, (0.12, 0.88, 0.40)),
                        (0.52, (0.92, 0.10, 0.06))):
        parts.append(mesh.make_ellipsoid(
            radii=(0.045, 0.025, 0.050), color=color,
            center=(8.36, 3.95, zoff), rings=4, segments=9,
            emissive=0.72, gloss=0.72,
        ))

    # Driver's suspension seat and armrests (visible when looking around).
    parts.append(mesh.make_rounded_box_x(
        0.82, 0.24, 0.92, (0.10, 0.16, 0.21),
        center=(7.82, 1.40, 0.0), chamfer=0.10, gloss=0.25,
    ))
    parts.append(mesh.make_rounded_panel(
        0.94, 1.48, 0.18, (0.10, 0.16, 0.21),
        center=(7.42, 2.08, 0.0), normal_axis="x",
        corner_radius=0.18, corner_segments=6, gloss=0.25,
    ))
    for side in (-1.0, 1.0):
        parts.append(mesh.make_tube_between(
            (7.70, 1.58, side * 0.52), (8.16, 1.82, side * 0.52),
            0.045, dark, segments=9, gloss=0.56,
        ))

    # Twin windscreen wipers inside the aperture.
    for side in (-1.0, 1.0):
        parts.append(mesh.make_tube_between(
            (front_low_x - 0.06, 2.30, side * 0.70),
            (front_high_x - 0.10, 3.42, side * 0.18),
            0.022, (0.025, 0.03, 0.035), segments=8, gloss=0.72,
        ))

    return mesh.combine(parts)
