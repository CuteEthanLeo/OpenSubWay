"""Tests for runtime game settings and the options-overlay text model."""

from __future__ import annotations

from opensubway.render import camera as cam
from opensubway.render import hud as hud_mod
from opensubway.settings import FOV_MAX, FOV_MIN, GameSettings
from opensubway.sim.simulation import Simulation


def test_defaults():
    s = GameSettings()
    assert s.show_menu is False
    assert s.hud_visible is True
    assert s.muted is False
    assert FOV_MIN <= s.fov <= FOV_MAX


def test_volume_clamps_and_unmutes():
    s = GameSettings()
    s.muted = True
    s.volume_up()                       # raising volume clears mute
    assert s.muted is False
    for _ in range(30):
        s.volume_up()
    assert s.volume == 1.0
    for _ in range(30):
        s.volume_down()
    assert s.volume == 0.0


def test_mute_zeroes_effective_volume():
    s = GameSettings()
    s.volume = 0.7
    assert s.effective_volume() == 0.7
    s.toggle_mute()
    assert s.effective_volume() == 0.0
    s.toggle_mute()
    assert s.effective_volume() == 0.7


def test_fov_clamps():
    s = GameSettings()
    for _ in range(30):
        s.fov_up()
    assert s.fov == FOV_MAX
    for _ in range(30):
        s.fov_down()
    assert s.fov == FOV_MIN


def test_camera_direct_select():
    c = cam.Camera(mode=cam.DRIVE)
    c.set_mode("orbit")
    assert c.mode == cam.ORBIT
    c.set_mode("chase")
    assert c.mode == cam.CHASE
    c.set_mode("nonsense")              # ignored, stays put
    assert c.mode == cam.CHASE


def test_overlay_documents_camera_switch():
    sim = Simulation()
    s = GameSettings()
    c = cam.Camera(mode=cam.DRIVE)
    lines = hud_mod.compose_settings_overlay(s, c, sim)
    text = "\n".join(t for t, _ in lines)
    assert "OPTIONS" in text
    # The overlay must tell the player how to switch cameras.
    assert "1 / 2 / 3" in text
    assert "cycle" in text.lower()
