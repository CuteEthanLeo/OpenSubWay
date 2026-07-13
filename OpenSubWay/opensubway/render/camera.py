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


def yaw_model(position: glm.vec3, forward: glm.vec3) -> glm.mat4:
    """Model matrix placing an object at ``position`` with local +X along ``forward``.

    ``forward`` is assumed to lie in the XZ plane (ground movement).
    """
    theta = math.atan2(-forward.z, forward.x)  # R_y maps +X -> (cos, 0, -sin)
    model = glm.translate(glm.mat4(1.0), position)
    return glm.rotate(model, theta, glm.vec3(0.0, 1.0, 0.0))


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

        # Explicit eye/look, used by drive & chase modes.
        self._eye = glm.vec3(0.0, 3.0, 0.0)
        self._look = glm.vec3(1.0, 3.0, 0.0)
        # Smoothed cab look-height offset for a natural feel.

    # ------------------------------------------------------------- mode switch
    def cycle_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]

    def set_mode(self, mode: str):
        if mode in MODES:
            self.mode = mode

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
        self.yaw += dyaw
        self.pitch = max(glm.radians(-85.0), min(glm.radians(85.0), self.pitch + dpitch))

    def zoom(self, factor: float):
        self.distance = max(3.0, min(400.0, self.distance * factor))

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
            self._look = self._eye + fwd * 20.0 - up * 2.0
        elif self.mode == CHASE:
            back = config.TRAIN_SIZE[0] / 2 + 9.0
            self._eye = position - fwd * back + up * 5.5
            self._look = position + fwd * 6.0 + up * 1.5

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
