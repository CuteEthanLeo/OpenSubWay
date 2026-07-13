"""Camera and matrix helpers (pyglm), with the Vulkan clip-space correction."""

from __future__ import annotations

import math

import glm

from .. import config

# Camera modes (cycled with the C key).
DRIVE = "drive"    # first-person: sitting in the cab, looking down the track
CHASE = "chase"    # third-person behind the train
ORBIT = "orbit"    # free orbit around the whole loop
MODES = [DRIVE, CHASE, ORBIT]

# Vulkan clip space differs from OpenGL: Y points down and depth is [0, 1].
# This correction matrix is pre-multiplied onto an OpenGL-style projection.
_CLIP = glm.mat4(
    glm.vec4(1.0, 0.0, 0.0, 0.0),
    glm.vec4(0.0, -1.0, 0.0, 0.0),
    glm.vec4(0.0, 0.0, 0.5, 0.0),
    glm.vec4(0.0, 0.0, 0.5, 1.0),
)


def mat4_bytes(m: glm.mat4) -> bytes:
    """Column-major float32 bytes for a mat4 (what Vulkan/GLSL expect).

    NOTE: pyglm's ``bytes(mat)`` serialises row-major, so we transpose first to
    hand Vulkan the column-major layout it reads.
    """
    return bytes(glm.transpose(m))


def yaw_model(position: glm.vec3, forward: glm.vec3, roll: float = 0.0) -> glm.mat4:
    """Rail-vehicle pose with grade, heading and optional curve cant.

    The legacy implementation discarded ``forward.y`` and therefore kept a
    car horizontal while it entered the tunnel ramp.  Building an orthonormal
    basis aligns local +X to the bogie chord and can bank local Y/Z about it.
    """
    x_axis = glm.normalize(forward)
    world_up = glm.vec3(0.0, 1.0, 0.0)
    if abs(glm.dot(x_axis, world_up)) > 0.98:
        world_up = glm.vec3(0.0, 0.0, 1.0)
    z_axis = glm.normalize(glm.cross(x_axis, world_up))
    y_axis = glm.normalize(glm.cross(z_axis, x_axis))
    if abs(roll) > 1e-8:
        cs, sn = math.cos(roll), math.sin(roll)
        rolled_y = y_axis * cs + z_axis * sn
        rolled_z = z_axis * cs - y_axis * sn
        y_axis, z_axis = rolled_y, rolled_z
    return glm.mat4(
        glm.vec4(x_axis, 0.0),
        glm.vec4(y_axis, 0.0),
        glm.vec4(z_axis, 0.0),
        glm.vec4(position, 1.0),
    )


def perspective(aspect: float, fov_degrees: float | None = None) -> glm.mat4:
    fov = config.FOV_DEGREES if fov_degrees is None else fov_degrees
    proj = glm.perspective(
        glm.radians(fov), aspect, config.NEAR_PLANE, config.FAR_PLANE
    )
    return _CLIP * proj


class Camera:
    """Multi-mode camera: first-person drive, third-person chase, free orbit."""

    def __init__(self, mode: str = DRIVE):
        self.mode = mode
        # Orbit parameters.
        self.target = glm.vec3(0.0, 0.0, 0.0)
        self.distance = 120.0
        self.yaw = glm.radians(45.0)
        self.pitch = glm.radians(30.0)
        # Mouse-look offsets used by drive and chase modes.
        self.look_yaw = glm.radians(-28.0) if mode == CHASE else 0.0
        self.look_pitch = glm.radians(9.0) if mode == CHASE else glm.radians(6.0)
        self.consist_length = (
            config.TRAIN_CARS * config.TRAIN_SIZE[0]
            + (config.TRAIN_CARS - 1) * config.TRAIN_CAR_GAP
        )
        self.chase_distance = self.consist_length * 0.49 + 90.0

        # Explicit eye/look, used by drive & chase modes.
        self._eye = glm.vec3(0.0, 3.0, 0.0)
        self._look = glm.vec3(1.0, 3.0, 0.0)
        # Smoothed cab look-height offset for a natural feel.

    # ------------------------------------------------------------- mode switch
    def cycle_mode(self):
        self.set_mode(MODES[(MODES.index(self.mode) + 1) % len(MODES)])

    def set_mode(self, mode: str):
        if mode in MODES:
            entering_chase = mode == CHASE and self.mode != CHASE
            self.mode = mode
            if entering_chase:
                self.look_yaw = glm.radians(-28.0)
                self.look_pitch = glm.radians(9.0)

    @property
    def is_drive(self) -> bool:
        return self.mode == DRIVE

    # ------------------------------------------------------------------ orbit
    def orbit_eye(self) -> glm.vec3:
        cp = glm.cos(self.pitch)
        return self.target + glm.vec3(
            self.distance * cp * glm.sin(self.yaw),
            self.distance * glm.sin(self.pitch),
            self.distance * cp * glm.cos(self.yaw),
        )

    def orbit(self, dyaw: float, dpitch: float):
        if self.mode == ORBIT:
            self.yaw += dyaw
            self.pitch = max(glm.radians(-85.0), min(glm.radians(85.0), self.pitch + dpitch))
        else:
            self.look_yaw += dyaw
            self.look_pitch = max(
                glm.radians(-70.0), min(glm.radians(70.0), self.look_pitch + dpitch)
            )

    def zoom(self, factor: float):
        if self.mode == ORBIT:
            self.distance = max(3.0, min(120000.0, self.distance * factor))
        elif self.mode == CHASE:
            minimum = self.consist_length * 0.49 + 55.0
            self.chase_distance = max(
                minimum, min(380.0, self.chase_distance * factor)
            )

    def reset_look(self):
        self.look_yaw = glm.radians(-28.0) if self.mode == CHASE else 0.0
        self.look_pitch = (glm.radians(9.0) if self.mode == CHASE
                           else glm.radians(6.0))

    def _look_direction(self, forward, pitch=None):
        up = glm.vec3(0.0, 1.0, 0.0)
        yawed = glm.vec3(glm.rotate(glm.mat4(1.0), self.look_yaw, up) * glm.vec4(forward, 0.0))
        angle = self.look_pitch if pitch is None else pitch
        horizontal = glm.normalize(glm.vec3(yawed.x, 0.0, yawed.z))
        return glm.normalize(horizontal * glm.cos(angle) + up * glm.sin(angle))

    # ---------------------------------------------------- per-frame placement
    def update_from_train(self, position: glm.vec3, forward: glm.vec3):
        """Position the camera relative to the train for drive / chase modes."""
        up = glm.vec3(0.0, 1.0, 0.0)
        fwd = glm.normalize(forward)
        if self.mode == DRIVE:
            # In the cab: eye just above and at the nose, looking down the track.
            cab_h = config.TRAIN_SIZE[1] * 0.75 + 1.2
            nose = config.TRAIN_SIZE[0] / 2 + 0.4
            self._eye = position + fwd * nose + up * cab_h
            # Free mouse-look while the cab remains attached to the train.
            look_dir = self._look_direction(fwd, self.look_pitch - glm.radians(10.0))
            self._look = self._eye + look_dir * 24.0
        elif self.mode == CHASE:
            # Orbit around the *centre of the complete consist*.  Looking at the
            # leading-car origin made a 101 m four-car train fill the view with
            # its tail roof and placed the camera inside the rear gangway.
            around = self._look_direction(fwd)
            horizontal = glm.normalize(glm.vec3(around.x, 0.0, around.z))
            midpoint = self.consist_length * 0.49
            radius = max(30.0, self.chase_distance - midpoint)
            vertical = max(3.5, min(30.0, 4.0 + radius * glm.tan(self.look_pitch)))
            self._look = position - fwd * midpoint + up * 2.25
            if position.y < -1.0:
                # Stay physically inside the tunnel bore.  A normal 360-degree
                # surface orbit would put the camera behind the concrete shell.
                side = glm.normalize(glm.vec3(-fwd.z, 0.0, fwd.x))
                lateral = config.OPPOSING_TRACK_OFFSET * 0.5
                tunnel_radius = min(radius, 65.0)
                # Eye at cab-window height, not against the tunnel crown; the
                # latter turned the entire 101 m blue/gold bodyside into a huge
                # perspective wedge across the ceiling.
                self._eye = (self._look - fwd * tunnel_radius + side * lateral
                             + up * 0.65)
            else:
                self._eye = self._look - horizontal * radius + up * vertical

    def view(self) -> glm.mat4:
        up = glm.vec3(0.0, 1.0, 0.0)
        if self.mode == ORBIT:
            return glm.lookAt(self.orbit_eye(), self.target, up)
        return glm.lookAt(self._eye, self._look, up)

    def eye_position(self) -> glm.vec3:
        """Camera world position (for lighting / fog)."""
        if self.mode == ORBIT:
            return self.orbit_eye()
        return self._eye
