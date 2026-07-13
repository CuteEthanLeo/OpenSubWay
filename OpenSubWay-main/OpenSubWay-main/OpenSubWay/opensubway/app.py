"""Application: window, main loop, world rendering, and the simulation."""

from __future__ import annotations

import os
import time

import glfw
import glm
from vulkan import vkDeviceWaitIdle

from . import config
from .audio.sound import SoundManager
from .input import InputState
from .settings import GameSettings
from .render import camera as cam
from .render import city as city_mod
from .render import hud as hud_mod
from .render import people as people_mod
from .render import worldmesh
from .render.panel import ControlPanel
from .sim.simulation import Simulation
from .vk.context import VulkanContext
from .vk.renderer import GpuMesh, Renderer


class App:
    def __init__(self):
        self._init_window()
        self.ctx = VulkanContext(self.window)
        self.renderer = Renderer(self.ctx)

        # --- Simulation ---
        start_station = int(os.environ.get("OPENSUBWAY_START_STATION", "0") or "0")
        self.sim = Simulation(start_index=start_station)
        self.line = self.sim.line
        self.stations = self.sim.stations

        # --- GPU geometry ---
        self.static_mesh = GpuMesh(
            self.ctx, worldmesh.build_static_mesh(self.line, self.stations)
        )
        self.city_mesh = GpuMesh(self.ctx, city_mod.build_city_mesh(self.line))
        self.train_meshes = [
            GpuMesh(self.ctx, worldmesh.build_train_mesh("lead")),
            GpuMesh(self.ctx, worldmesh.build_train_mesh("middle")),
            GpuMesh(self.ctx, worldmesh.build_train_mesh("tail")),
        ]
        # Compatibility alias for code that only needs the leading car mesh.
        self.train_mesh = self.train_meshes[0]
        self.person_mesh = GpuMesh(self.ctx, people_mod.build_person_mesh())
        self.signal_meshes = {
            "red": GpuMesh(self.ctx, worldmesh.build_signal_lamp_mesh((1.0, 0.035, 0.02))),
            "yellow": GpuMesh(self.ctx, worldmesh.build_signal_lamp_mesh((1.0, 0.62, 0.04))),
            "green": GpuMesh(self.ctx, worldmesh.build_signal_lamp_mesh((0.04, 1.0, 0.18))),
        }
        self.signal_sites = worldmesh.signal_sites(self.line)
        self._gpu_meshes = [self.static_mesh, self.city_mesh, *self.train_meshes,
                            self.person_mesh, *self.signal_meshes.values()]

        # --- Camera: default to the first-person driver's cab view ---
        start_mode = os.environ.get("OPENSUBWAY_CAMERA", cam.DRIVE).lower()
        if start_mode not in cam.MODES:
            start_mode = cam.DRIVE
        self.camera = cam.Camera(mode=start_mode)
        centroid = self.line.centroid()
        span = max(glm.length(p - centroid) for p in self.line.points)
        self.camera.target = centroid + glm.vec3(0.0, 6.0, 0.0)
        self.camera.distance = span * 2.1 + 40.0
        _od = os.environ.get("OPENSUBWAY_ORBIT_DIST")
        if _od:
            si = int(os.environ.get("OPENSUBWAY_ORBIT_STATION", "0"))
            self.camera.distance = float(_od)
            self.camera.target = self.stations[si].position + glm.vec3(0.0, 2.0, 0.0)
            self.camera.pitch = glm.radians(float(os.environ.get("OPENSUBWAY_ORBIT_PITCH", "10")))
            self.camera.yaw = glm.radians(float(os.environ.get("OPENSUBWAY_ORBIT_YAW", "45")))

        self.panel = ControlPanel(self.sim.train)
        self.input = InputState(self.window, self.camera, self.renderer, self, self.panel)

        # --- Player settings (camera/audio/FOV/HUD, tweaked live) ---
        self.settings = GameSettings()
        self.camera.fov = self.settings.fov
        if os.environ.get("OPENSUBWAY_OPEN_MENU"):
            self.settings.show_menu = True   # demo/screenshot the options overlay

        # --- Audio ---
        self.sound = SoundManager(enabled=os.environ.get("OPENSUBWAY_NOAUDIO") is None)
        self.sound.set_master(self.settings.effective_volume())

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialise GLFW.")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        self.window = glfw.create_window(
            config.WINDOW_WIDTH, config.WINDOW_HEIGHT, config.WINDOW_TITLE, None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window.")

    # ------------------------------------------------------------- controls
    def toggle_pause(self):
        self.sim.paused = not self.sim.paused

    def reset(self):
        self.sim.reset()
        self.panel.train = self.sim.train   # panel points at the fresh train

    def horn(self):
        self.sim.train.horn()

    # --- settings / options menu ---
    def toggle_settings(self):
        self.settings.show_menu = not self.settings.show_menu

    def toggle_hud(self):
        self.settings.hud_visible = not self.settings.hud_visible

    def set_camera(self, mode):
        self.camera.set_mode(mode)

    def toggle_mute(self):
        self.settings.toggle_mute()
        self.sound.set_master(self.settings.effective_volume())

    def volume_up(self):
        self.settings.volume_up()
        self.sound.set_master(self.settings.effective_volume())

    def volume_down(self):
        self.settings.volume_down()
        self.sound.set_master(self.settings.effective_volume())

    def fov_up(self):
        self.settings.fov_up()
        self.camera.fov = self.settings.fov

    def fov_down(self):
        self.settings.fov_down()
        self.camera.fov = self.settings.fov

    # ----------------------------------------------------------------- loop
    def run(self):
        autoclose = float(os.environ.get("OPENSUBWAY_AUTOCLOSE", "0") or "0")
        start = time.perf_counter()
        last = time.perf_counter()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if autoclose and (time.perf_counter() - start) >= autoclose:
                break

            now = time.perf_counter()
            dt = min(0.1, now - last)
            last = now
            self.sim.update(dt)

            # Audio: speed-driven rumble + one-shot events.
            self.sound.update(self.sim.train.speed)
            for ev in self.sim.frame_events:
                self.sound.event(ev)

            player_cars = self.sim.consist_placements(opposite=False)
            opposing_services = [
                self.sim.consist_placements(opposite=True, service_index=i)
                for i in range(len(self.sim.opposing_distances))
            ]
            train_model, train_pos, train_fwd = player_cars[0]
            self.camera.update_from_train(train_pos, train_fwd)

            items = [
                (self.static_mesh, glm.mat4(1.0)),
                (self.city_mesh, glm.mat4(1.0)),
            ]
            # Passengers on the platforms.
            for pos, bob in self.sim.passengers.instances():
                model = glm.translate(glm.mat4(1.0), glm.vec3(pos.x, pos.y + bob, pos.z))
                items.append((self.person_mesh, model))
            # Hide the player's consist in cab view; the opposing train remains visible.
            if not self.camera.is_drive:
                for car_index, (model, _, _) in enumerate(player_cars):
                    mesh_index = (0 if car_index == 0 else
                                  2 if car_index == len(player_cars) - 1 else 1)
                    items.append((self.train_meshes[mesh_index], model))
            for opposing_cars in opposing_services:
                for car_index, (model, _, _) in enumerate(opposing_cars):
                    mesh_index = (0 if car_index == 0 else
                                  2 if car_index == len(opposing_cars) - 1 else 1)
                    items.append((self.train_meshes[mesh_index], model))

            # Live three-aspect block signals on both running tracks.
            aspect_y = {"red": 4.10, "yellow": 3.72, "green": 3.34}
            for site in self.signal_sites:
                state = self.sim.signal_state(site["track"], site["distance"])
                p = site["post"] - site["direction"] * 0.20
                model = glm.translate(
                    glm.mat4(1.0), glm.vec3(p.x, p.y + aspect_y[state], p.z)
                )
                items.append((self.signal_meshes[state], model))

            hud_lines = (hud_mod.compose_lines(self.sim, self.camera)
                         if self.settings.hud_visible else None)
            overlay = (hud_mod.compose_settings_overlay(self.settings, self.camera, self.sim)
                       if self.settings.show_menu else None)
            ext = self.renderer.swap.extent
            buttons = self.panel.layout(ext.width, ext.height)
            upcoming = [s for s in self.signal_sites
                        if s["track"] == "main"
                        and s["distance"] >= self.sim.train.distance]
            next_signal = min(upcoming, key=lambda s: s["distance"], default=None)
            signal_state = (self.sim.signal_state("main", next_signal["distance"])
                            if next_signal else "red")
            telemetry = {
                "speed": self.sim.train.speed_kmh(),
                "limit": self.sim.speed_limit_kmh(),
                "station": self.sim.train.current_station_name(),
                "next": self.sim.train.next_station_name(),
                "throttle": self.sim.train.throttle_notch / config.THROTTLE_NOTCHES,
                "brake": self.sim.train.brake_notch / config.BRAKE_NOTCHES,
                "signal": signal_state,
                "signal_distance": ((next_signal["distance"] - self.sim.train.distance)
                                    if next_signal else 0.0),
                "dispatch": self.line.stations[
                    self.line.nearest_station_index(self.sim.opposing_distance)
                ].name,
                "distance": self.sim.train.distance,
                "route_stations": [
                    (station.distance, station.underground)
                    for station in self.line.stations
                ],
                "ai_distances": list(self.sim.opposing_distances),
            }
            self.renderer.draw(
                self.camera, items, hud_lines, buttons, overlay, telemetry=telemetry
            )

        elapsed = time.perf_counter() - start
        if elapsed > 0:
            print(f"Avg FPS: {self.renderer.frames_presented / elapsed:.1f} "
                  f"({self.renderer.frames_presented} frames / {elapsed:.1f}s)")
        self._shutdown()

    def _shutdown(self):
        self.sound.shutdown()
        vkDeviceWaitIdle(self.ctx.device)
        for gm in self._gpu_meshes:
            gm.destroy()
        self.renderer.destroy()
        self.ctx.destroy()
        glfw.destroy_window(self.window)
        glfw.terminate()
