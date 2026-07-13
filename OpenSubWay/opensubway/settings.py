"""Runtime game settings (adjusted live from the keyboard / options menu)."""

from __future__ import annotations

from . import config

# Field-of-view limits for the FOV setting (degrees).
FOV_MIN = 45.0
FOV_MAX = 90.0
FOV_STEP = 5.0

# Master-volume step per key press.
VOLUME_STEP = 0.1


class GameSettings:
    """Player-tweakable options, separate from the fixed engine config.

    App owns one of these and applies changes to the camera / sound / sim.
    Everything here is safe to change every frame.
    """

    def __init__(self):
        self.show_menu = False          # Options overlay visible (TAB)
        self.hud_visible = True         # top-left status/objectives panel (F1)
        self.muted = False              # audio muted (M)
        self.volume = 0.8               # master volume 0..1 (- / =)
        self.fov = float(config.FOV_DEGREES)  # perspective FOV in degrees ([ / ])

    # -- audio ----------------------------------------------------------
    def effective_volume(self) -> float:
        return 0.0 if self.muted else self.volume

    def volume_up(self):
        self.volume = min(1.0, round(self.volume + VOLUME_STEP, 3))
        self.muted = False

    def volume_down(self):
        self.volume = max(0.0, round(self.volume - VOLUME_STEP, 3))

    def toggle_mute(self):
        self.muted = not self.muted

    # -- field of view --------------------------------------------------
    def fov_up(self):
        self.fov = min(FOV_MAX, self.fov + FOV_STEP)

    def fov_down(self):
        self.fov = max(FOV_MIN, self.fov - FOV_STEP)
