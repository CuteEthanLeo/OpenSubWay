"""Airport-line traffic, camera and consist regressions."""

from __future__ import annotations

import glm

from opensubway import config
from opensubway.render import worldmesh
from opensubway.render.camera import Camera, CHASE, DRIVE
from opensubway.sim.simulation import GREEN, RED, YELLOW, Simulation


def test_distinct_high_detail_train_car_meshes_build():
    lead = worldmesh.build_train_mesh("lead")
    middle = worldmesh.build_train_mesh("middle")
    tail = worldmesh.build_train_mesh("tail")
    assert lead.vertices.shape[0] > 6000
    assert middle.vertices[:, 1].max() <= 6.25
    assert tail.vertices.shape == lead.vertices.shape


def test_three_car_player_and_opposing_consists_use_separate_tracks():
    sim = Simulation()
    player = sim.consist_placements(False)
    opposing = sim.consist_placements(True)
    assert len(player) == config.TRAIN_CARS == 4
    assert len(opposing) == config.TRAIN_CARS
    assert glm.length(player[0][1] - opposing[0][1]) > 5.0
    assert glm.dot(player[0][2], opposing[0][2]) < -0.95


def test_dispatch_uses_realistic_headway_and_three_westbound_services():
    sim = Simulation()
    assert len(sim.opposing_distances) == 3
    gaps = [sim.opposing_distances[i] - sim.opposing_distances[i + 1]
            for i in range(len(sim.opposing_distances) - 1)]
    expected = config.OPPOSING_TRAIN_SPEED * config.AIRPORT_HEADWAY_SECONDS
    assert all(abs(gap - expected) < 1.0 for gap in gaps)


def test_speed_profile_has_terminal_and_line_speed_limits():
    sim = Simulation()
    assert sim.speed_limit_kmh() == 40.0
    sim.train.distance = (sim.stations[2].distance + sim.stations[3].distance) * 0.5
    assert sim.speed_limit_kmh() == 160.0


def test_atp_brakes_when_driver_exceeds_section_limit():
    sim = Simulation()
    train = sim.train
    train.speed_limit_kmh = 40.0
    train.speed = 20.0
    train.doors_open = False
    train.brake_notch = 0
    train.throttle_notch = 0
    before = train.speed
    train.update(0.1)
    assert train.atp_active
    assert train.speed < before


def test_block_signal_changes_as_train_enters_blocks():
    sim = Simulation()
    signal_d = sim.train.distance + 5.0
    sim.train.distance = signal_d + config.SIGNAL_BLOCK_LENGTH * 0.5
    assert sim.signal_state("main", signal_d) == RED
    sim.train.distance = signal_d + config.SIGNAL_BLOCK_LENGTH * 1.5
    assert sim.signal_state("main", signal_d) == YELLOW
    sim.train.distance = signal_d - 1.0
    assert sim.signal_state("main", signal_d) == GREEN


def test_drive_and_chase_mouse_drag_changes_camera_view():
    position = glm.vec3(100.0, 0.0, 0.0)
    forward = glm.vec3(1.0, 0.0, 0.0)
    camera = Camera(DRIVE)
    camera.update_from_train(position, forward)
    before = glm.vec3(camera._look)
    camera.orbit(0.45, 0.20)
    camera.update_from_train(position, forward)
    assert glm.length(camera._look - before) > 1.0

    camera.set_mode(CHASE)
    camera.update_from_train(position, forward)
    chase_before = glm.vec3(camera._eye)
    camera.orbit(0.4, -0.1)
    camera.update_from_train(position, forward)
    assert glm.length(camera._eye - chase_before) > 1.0
