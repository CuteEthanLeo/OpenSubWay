"""Track geometry regression checks."""

from __future__ import annotations

import glm

from opensubway.render import mesh


def test_extruded_rail_has_real_vertical_extent_and_side_normals():
    points = [glm.vec3(0, 0, 0), glm.vec3(5, 0, 0), glm.vec3(5, 0, 5)]
    rail = mesh.make_extruded_ribbon(
        points, 0.1, 0.2, 0.5, (0.5, 0.5, 0.5), closed=False,
    )
    assert abs(float(rail.vertices[:, 1].min()) - 0.2) < 1e-6
    assert abs(float(rail.vertices[:, 1].max()) - 0.5) < 1e-6
    assert any(abs(float(n[1])) < 0.1 for n in rail.vertices[:, 3:6])


def test_oriented_sleeper_rotates_long_axis_across_track():
    sleeper = mesh.make_oriented_box(
        (4.0, 0.2, 0.3), (0.3, 0.2, 0.1), (0, 0.1, 0), glm.vec3(0, 0, 1),
    )
    x_span = float(sleeper.vertices[:, 0].max() - sleeper.vertices[:, 0].min())
    z_span = float(sleeper.vertices[:, 2].max() - sleeper.vertices[:, 2].min())
    assert z_span > x_span * 5.0
