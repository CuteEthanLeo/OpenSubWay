"""Build the static world mesh (ground, track, stations) and the train mesh."""

from __future__ import annotations

import random

import glm

from .. import config
from . import mesh
from .mesh import Mesh


TRACK_BED_COLOR = (0.46, 0.41, 0.36)     # sunlit ballast gravel
VERGE_COLOR = (0.34, 0.44, 0.21)         # grass strip beside the ballast
PLATFORM_COLOR = (0.62, 0.62, 0.66)
BUILDING_COLOR = (0.72, 0.70, 0.66)
MAST_COLOR = (0.38, 0.40, 0.42)
WIRE_COLOR = (0.07, 0.07, 0.08)

SCENERY_SEED = 4321
CATENARY_SPACING = 22.0   # metres of arc between masts
TREE_SPACING = 8.0        # average metres of arc between line-side trees
WIRE_HEIGHT = 6.2


def _tree(parts, x, z, rng):
    """Low-poly tree: brown trunk plus two stacked green canopy boxes."""
    trunk_h = rng.uniform(1.0, 1.6)
    h = rng.uniform(2.8, 4.6)
    g = rng.uniform(-0.04, 0.04)
    green = (0.16 + g, 0.34 + rng.uniform(-0.04, 0.08), 0.11 + g)
    parts.append(
        mesh.make_box(size=(0.35, trunk_h, 0.35), color=(0.28, 0.20, 0.12),
                      center=(x, trunk_h / 2, z))
    )
    w = rng.uniform(1.8, 3.0)
    # Canopies are translucent so the low sun bleeds through the foliage.
    parts.append(
        mesh.make_box(size=(w, h * 0.6, w), color=green,
                      center=(x, trunk_h + h * 0.30, z), translucency=0.55)
    )
    top = (min(1.0, green[0] * 1.3), min(1.0, green[1] * 1.25), green[2] * 1.2)
    parts.append(
        mesh.make_box(size=(w * 0.62, h * 0.45, w * 0.62), color=top,
                      center=(x, trunk_h + h * 0.62, z), translucency=0.75)
    )


def _near_station_arc(line, d, margin):
    """True if arc-distance ``d`` lies within ``margin`` of any station node."""
    for s in line.stations:
        delta = abs((d - s.distance + line.total / 2) % line.total - line.total / 2)
        if delta <= margin:
            return True
    return False


def _catenary(parts, line):
    """Masts beside the track and a contact wire above the centreline."""
    d = 0.0
    while d < line.total:
        pos, fwd = line.position_at(d)
        perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
        base = pos - perp * (config.RAIL_HALF_WIDTH + 2.4)
        parts.append(
            mesh.make_box(size=(0.22, WIRE_HEIGHT + 0.6, 0.22), color=MAST_COLOR,
                          center=(base.x, (WIRE_HEIGHT + 0.6) / 2, base.z), gloss=0.45)
        )
        # Cantilever arm: a thin horizontal quad from the mast over the track.
        a = glm.vec3(base.x, WIRE_HEIGHT + 0.25, base.z)
        b = glm.vec3(pos.x, WIRE_HEIGHT + 0.25, pos.z)
        w = glm.vec3(fwd.x, 0.0, fwd.z) * 0.09
        corners = [
            (a.x - w.x, a.y, a.z - w.z),
            (a.x + w.x, a.y, a.z + w.z),
            (b.x + w.x, b.y, b.z + w.z),
            (b.x - w.x, b.y, b.z - w.z),
        ]
        parts.append(mesh.make_quad(corners, (0.0, 1.0, 0.0), MAST_COLOR, 0.0, gloss=0.45))
        d += CATENARY_SPACING

    # The contact wire itself: a narrow ribbon following the whole loop.
    parts.append(mesh.make_loop_ribbon(line.points, 0.05, WIRE_HEIGHT, WIRE_COLOR, gloss=0.6))


def _lineside_trees(parts, line, rng):
    """Greenery along the corridor, clear of platforms and buildings."""
    d = rng.uniform(0.0, TREE_SPACING)
    while d < line.total:
        step = rng.uniform(TREE_SPACING * 0.6, TREE_SPACING * 1.6)
        if not _near_station_arc(line, d, 14.0):
            pos, fwd = line.position_at(d)
            perp = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
            side = 1.0 if rng.random() < 0.5 else -1.0
            off = rng.uniform(9.0, 15.0)
            p = pos + perp * (side * off)
            _tree(parts, p.x + rng.uniform(-1.5, 1.5), p.z + rng.uniform(-1.5, 1.5), rng)
        d += step


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

    # Grass verge, ballast bed, and the two rails following the looped line.
    parts.append(mesh.make_loop_ribbon(line.points, 7.5, 0.02, VERGE_COLOR))
    parts.append(
        mesh.make_loop_ribbon(line.points, config.RAIL_HALF_WIDTH + 0.8, 0.06, TRACK_BED_COLOR)
    )
    left = mesh.offset_loop(line.points, config.RAIL_HALF_WIDTH)
    right = mesh.offset_loop(line.points, -config.RAIL_HALF_WIDTH)
    # Polished railheads catch the low sun.
    parts.append(mesh.make_loop_ribbon(left, 0.18, 0.22, config.RAIL_COLOR, gloss=0.9))
    parts.append(mesh.make_loop_ribbon(right, 0.18, 0.22, config.RAIL_COLOR, gloss=0.9))

    # Electrification and line-side greenery (like the reference scenery).
    _catenary(parts, line)
    _lineside_trees(parts, line, rng)

    # Stations: a platform beside the track, a building, and a coloured roof.
    perps = mesh._loop_perps(line.points)
    sx, sy, sz = config.STATION_SIZE
    for i, station in enumerate(stations):
        node = line.points[i]
        perp = perps[i]
        tangent = glm.normalize(glm.vec3(perp.z, 0.0, -perp.x))
        platform_off = config.RAIL_HALF_WIDTH + 2.0
        building_off = platform_off + sz / 2 + 1.0

        platform_c = node + perp * platform_off
        parts.append(
            mesh.make_box(
                size=(sx, 0.5, 4.0),
                color=PLATFORM_COLOR,
                center=(platform_c.x, 0.25, platform_c.z),
                gloss=0.15,
            )
        )
        # Lamp posts along the platform (off during the day).
        for t in (-sx * 0.4, 0.0, sx * 0.4):
            lp = platform_c + tangent * t + perp * 1.5
            parts.append(
                mesh.make_box(size=(0.25, 4.0, 0.25), color=(0.30, 0.31, 0.33),
                              center=(lp.x, 2.0, lp.z), gloss=0.4)
            )
            parts.append(
                mesh.make_box(size=(0.9, 0.5, 0.9), color=(0.85, 0.88, 0.92),
                              center=(lp.x, 4.2, lp.z), gloss=0.5)
            )

        building_c = node + perp * building_off
        parts.append(
            mesh.make_box(
                size=(sx, sy, sz),
                color=BUILDING_COLOR,
                center=(building_c.x, sy / 2, building_c.z),
                gloss=0.18,
            )
        )
        # Emissive station-colour sign band on the roof.
        parts.append(
            mesh.make_box(
                size=(sx * 1.08, 0.8, sz * 1.08),
                color=station.color,
                center=(building_c.x, sy + 0.4, building_c.z),
                emissive=0.55,
                gloss=0.3,
            )
        )

    return mesh.combine(parts)


def build_train_mesh() -> Mesh:
    """Single car with local +X as forward; sits just above the rails.

    Styled after a modern white EMU: dark window band, green accent stripe,
    grey roof, and emissive headlights / destination sign at the +X end.
    """
    length, height, width = config.TRAIN_SIZE
    base_y = height / 2 + 0.35
    parts = [
        # Polished paintwork catches the low sun along the bodyside.
        mesh.make_box(
            size=(length, height, width), color=config.TRAIN_COLOR,
            center=(0.0, base_y, 0.0), gloss=0.85,
        ),
        # Dark tinted window band along the upper body (glass).
        mesh.make_box(
            size=(length * 0.90, height * 0.30, width * 1.02),
            color=(0.09, 0.10, 0.12),
            center=(0.0, base_y + height * 0.16, 0.0),
            gloss=0.95,
        ),
        # Green accent stripe below the windows.
        mesh.make_box(
            size=(length * 1.005, height * 0.13, width * 1.005),
            color=(0.33, 0.62, 0.24),
            center=(0.0, base_y - height * 0.08, 0.0),
            gloss=0.8,
        ),
        # Grey equipment roof (matte).
        mesh.make_box(
            size=(length * 0.96, 0.5, width * 0.90),
            color=(0.44, 0.46, 0.48),
            center=(0.0, height + 0.35, 0.0),
            gloss=0.25,
        ),
        # Destination sign above the cab window (emissive, reads at day).
        mesh.make_box(
            size=(0.15, 0.45, width * 0.5),
            color=(1.0, 0.72, 0.15),
            center=(length / 2 + 0.02, base_y + height * 0.30, 0.0),
            emissive=0.9,
        ),
    ]
    # Twin headlights near the base of the cab front.
    for zoff in (-width * 0.30, width * 0.30):
        parts.append(
            mesh.make_box(
                size=(0.12, 0.28, 0.40),
                color=(1.0, 0.98, 0.90),
                center=(length / 2 + 0.02, base_y - height * 0.28, zoff),
                emissive=1.0,
            )
        )
    return mesh.combine(parts)
