"""Sound playback via pygame.mixer. Degrades gracefully with no audio device."""

from __future__ import annotations

from .. import config


class SoundManager:
    AMBIENCE_BASE = 0.35   # ambience bed volume at master 1.0

    def __init__(self, enabled: bool = True):
        self.ok = False
        self._pygame = None
        self._rumble_ch = None
        self._amb_ch = None
        self._master = 1.0     # 0..1 master multiplier (see set_master)
        self._speed_vol = 0.0  # last speed-driven rumble level (pre-master)
        if not enabled:
            return
        try:
            import pygame

            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            self._pygame = pygame
            self._build_sounds()
            # Start the looping beds.
            self._amb_ch = self._ambience.play(loops=-1)
            self._rumble_ch = self._rumble.play(loops=-1)
            if self._amb_ch:
                self._amb_ch.set_volume(self.AMBIENCE_BASE)
            if self._rumble_ch:
                self._rumble_ch.set_volume(0.0)
            self.ok = True
            print("Audio: enabled")
        except Exception as exc:  # no device, driver, etc.
            print(f"Audio: disabled ({exc})")

    def _build_sounds(self):
        from . import synth

        mk = self._pygame.sndarray.make_sound
        self._rumble = mk(synth.rumble())
        self._ambience = mk(synth.ambience())
        self._chime = mk(synth.chime())
        self._horn = mk(synth.horn())

    def set_master(self, master: float):
        """Set the master-volume multiplier (0 = silent) and re-apply it."""
        self._master = max(0.0, min(1.0, master))
        if not self.ok:
            return
        if self._amb_ch:
            self._amb_ch.set_volume(self.AMBIENCE_BASE * self._master)
        if self._rumble_ch:
            self._rumble_ch.set_volume(self._speed_vol * self._master)

    def update(self, speed: float):
        """Modulate the rumble bed by train speed (scaled by master volume)."""
        if not self.ok or self._rumble_ch is None:
            return
        frac = min(1.0, abs(speed) / config.TRAIN_MAX_SPEED)
        self._speed_vol = 0.0 if abs(speed) < config.STOP_EPSILON else 0.18 + 0.6 * frac
        self._rumble_ch.set_volume(self._speed_vol * self._master)

    def event(self, name: str):
        if not self.ok:
            return
        if name in ("doors", "arrive", "depart"):
            self._chime.set_volume(self._master)
            self._chime.play()
        elif name == "horn":
            self._horn.set_volume(self._master)
            self._horn.play()

    def shutdown(self):
        if self.ok and self._pygame is not None:
            try:
                self._pygame.mixer.stop()
                self._pygame.mixer.quit()
            except Exception:
                pass
