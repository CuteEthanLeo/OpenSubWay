"""Procedurally synthesized sound effects (numpy) — no audio asset files.

All generators return C-contiguous stereo int16 arrays at 44.1 kHz, ready for
``pygame.sndarray.make_sound``. Looping sounds use frequencies that are integer
multiples of 1/duration so the loop points are seamless.
"""

from __future__ import annotations

import numpy as np

SR = 44100


def _stereo(mono: np.ndarray, volume: float = 1.0) -> np.ndarray:
    mono = np.clip(mono * volume, -1.0, 1.0)
    i16 = (mono * 32767).astype(np.int16)
    return np.ascontiguousarray(np.column_stack([i16, i16]))


def rumble(dur: float = 2.0) -> np.ndarray:
    """Low motor/rail rumble, seamless loop."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    rng = np.random.default_rng(1)
    # Tonal low end (integer cycles over dur => seamless).
    base = 0.55 * np.sin(2 * np.pi * 45 * t) + 0.32 * np.sin(2 * np.pi * 67 * t)
    base += 0.18 * np.sin(2 * np.pi * 90 * t)
    # Textured noise, low-passed with a moving average.
    noise = rng.standard_normal(n)
    k = 180
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    sig = 0.8 * base + 1.4 * noise
    sig /= np.max(np.abs(sig)) + 1e-6
    return _stereo(sig, 0.9)


def ambience(dur: float = 4.0) -> np.ndarray:
    """Quiet night-city hum, seamless loop."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    hum = 0.4 * np.sin(2 * np.pi * 60 * t) + 0.25 * np.sin(2 * np.pi * 120 * t)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(400) / 400, mode="same")
    sig = 0.5 * hum + 1.0 * noise
    sig /= np.max(np.abs(sig)) + 1e-6
    return _stereo(sig, 0.5)


def chime(dur: float = 0.55) -> np.ndarray:
    """Two-tone station/door chime with a soft decay."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    env = np.exp(-4.5 * t)
    half = n // 2
    tone = np.zeros(n)
    tone[:half] += np.sin(2 * np.pi * 784 * t[:half])   # G5
    tone[half:] += np.sin(2 * np.pi * 1047 * t[half:])  # C6
    return _stereo(tone * env, 0.6)


def horn(dur: float = 0.9) -> np.ndarray:
    """A two-note detuned horn blast."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    env = np.minimum(1.0, 12 * t) * np.minimum(1.0, 12 * (dur - t))
    tone = (
        np.sin(2 * np.pi * 165 * t)
        + 0.7 * np.sin(2 * np.pi * 208 * t)
        + 0.3 * np.sin(2 * np.pi * 330 * t)
    )
    tone /= np.max(np.abs(tone)) + 1e-6
    return _stereo(tone * env, 0.7)
