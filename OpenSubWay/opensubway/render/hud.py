"""Heads-up display: font atlas (Pillow) + text/panel geometry for the overlay.

The atlas is a single-channel (R8) image of ASCII glyphs plus one solid cell
used to draw the translucent panel background. Vertices are emitted directly in
normalized device coordinates (pos.xy, uv, rgba) for the text pipeline.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..render import camera as cam

FIRST_CHAR = 32
LAST_CHAR = 126
COLS = 16
FONT_SIZE = 26

MARGIN = 18
PAD = 12

# Colors (r, g, b, a)
WHITE = (0.92, 0.94, 0.98, 1.0)
GRAY = (0.66, 0.68, 0.74, 1.0)
GREEN = (0.40, 0.90, 0.52, 1.0)
YELLOW = (0.98, 0.86, 0.35, 1.0)
CYAN = (0.55, 0.85, 0.95, 1.0)
PANEL = (0.05, 0.06, 0.09, 0.62)


def _load_font():
    for path in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


class FontAtlas:
    """Rasterised ASCII glyph atlas with per-char UV rectangles."""

    def __init__(self):
        font = _load_font()
        ascent, descent = font.getmetrics()
        self.cell_h = ascent + descent
        try:
            self.cell_w = max(1, math.ceil(font.getlength("M")))
        except Exception:
            self.cell_w = FONT_SIZE // 2 + 2

        chars = [chr(c) for c in range(FIRST_CHAR, LAST_CHAR + 1)]
        count = len(chars) + 1  # + solid cell
        rows = math.ceil(count / COLS)
        self.width = COLS * self.cell_w
        self.height = rows * self.cell_h

        img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        self.char_uv: dict[str, tuple] = {}
        for i, ch in enumerate(chars):
            cx = (i % COLS) * self.cell_w
            cy = (i // COLS) * self.cell_h
            draw.text((cx, cy), ch, fill=255, font=font)
            self.char_uv[ch] = (
                cx / self.width, cy / self.height,
                (cx + self.cell_w) / self.width, (cy + self.cell_h) / self.height,
            )

        # Solid cell for panel/background quads.
        si = len(chars)
        scx = (si % COLS) * self.cell_w
        scy = (si // COLS) * self.cell_h
        draw.rectangle([scx, scy, scx + self.cell_w - 1, scy + self.cell_h - 1], fill=255)
        self.solid_uv = ((scx + self.cell_w * 0.5) / self.width,
                         (scy + self.cell_h * 0.5) / self.height)

        self.pixels = img.tobytes()  # L -> 1 byte per pixel == R8


class HudBuilder:
    """Turns HUD content into a vertex buffer for the text pipeline."""

    def __init__(self, atlas: FontAtlas):
        self.atlas = atlas

    # -- low level -------------------------------------------------------
    @staticmethod
    def _ndc(px, py, sw, sh):
        return (px / sw) * 2.0 - 1.0, (py / sh) * 2.0 - 1.0

    def _push_quad(self, verts, x0, y0, x1, y1, uvs, color, sw, sh):
        (u0, v0, u1, v1) = uvs
        nx0, ny0 = self._ndc(x0, y0, sw, sh)
        nx1, ny1 = self._ndc(x1, y1, sw, sh)
        r, g, b, a = color
        tl = [nx0, ny0, u0, v0, r, g, b, a]
        tr = [nx1, ny0, u1, v0, r, g, b, a]
        br = [nx1, ny1, u1, v1, r, g, b, a]
        bl = [nx0, ny1, u0, v1, r, g, b, a]
        verts += tl + tr + br + tl + br + bl

    def _push_solid(self, verts, x0, y0, x1, y1, color, sw, sh):
        u, v = self.atlas.solid_uv
        self._push_quad(verts, x0, y0, x1, y1, (u, v, u, v), color, sw, sh)

    def _emit_text(self, verts, text, x, y, color, sw, sh):
        gw, gh = self.atlas.cell_w, self.atlas.cell_h
        tx = x
        for ch in text:
            uv = self.atlas.char_uv.get(ch)
            if uv and ch != " ":
                self._push_quad(verts, tx, y, tx + gw, y + gh, uv, color, sw, sh)
            tx += gw

    def _emit_hud(self, verts, lines, sw, sh):
        gw, gh = self.atlas.cell_w, self.atlas.cell_h
        max_len = max((len(t) for t, _ in lines), default=0)
        panel_w = max_len * gw + 2 * PAD
        panel_h = len(lines) * gh + 2 * PAD
        x0, y0 = MARGIN, MARGIN
        self._push_solid(verts, x0, y0, x0 + panel_w, y0 + panel_h, PANEL, sw, sh)
        ty = y0 + PAD
        for text, color in lines:
            self._emit_text(verts, text, x0 + PAD, ty, color, sw, sh)
            ty += gh

    def _emit_panel(self, verts, buttons, sw, sh):
        gw, gh = self.atlas.cell_w, self.atlas.cell_h
        if not buttons:
            return
        # Backing bar behind the whole button row.
        bx0 = min(b.x0 for b in buttons) - 8
        bx1 = max(b.x1 for b in buttons) + 8
        by0 = min(b.y0 for b in buttons) - 8
        by1 = max(b.y1 for b in buttons) + 8
        self._push_solid(verts, bx0, by0, bx1, by1, (0.10, 0.12, 0.17, 0.88), sw, sh)
        for b in buttons:
            base = b.tint if b.tint else (0.26, 0.30, 0.40)
            if b.active and b.tint is None:
                base = (0.30, 0.48, 0.68)
            col = (base[0], base[1], base[2], 0.98)
            self._push_solid(verts, b.x0, b.y0, b.x1, b.y1, col, sw, sh)
            tw = len(b.label) * gw
            tx = (b.x0 + b.x1) / 2 - tw / 2
            ty = (b.y0 + b.y1) / 2 - gh / 2
            self._emit_text(verts, b.label, tx, ty, WHITE, sw, sh)

    def _emit_overlay(self, verts, lines, sw, sh):
        """Centered modal panel (the Options menu), dimming the scene behind it."""
        gw, gh = self.atlas.cell_w, self.atlas.cell_h
        # Dim the whole screen so the menu reads as a modal.
        self._push_solid(verts, 0, 0, sw, sh, (0.0, 0.0, 0.0, 0.45), sw, sh)
        max_len = max((len(t) for t, _ in lines), default=0)
        panel_w = max_len * gw + 4 * PAD
        panel_h = len(lines) * gh + 4 * PAD
        x0 = (sw - panel_w) / 2
        y0 = (sh - panel_h) / 2
        self._push_solid(verts, x0, y0, x0 + panel_w, y0 + panel_h,
                         (0.06, 0.08, 0.12, 0.92), sw, sh)
        # Thin accent border along the top of the panel.
        self._push_solid(verts, x0, y0, x0 + panel_w, y0 + 4,
                         (0.98, 0.86, 0.35, 0.95), sw, sh)
        tx = x0 + 2 * PAD
        ty = y0 + 2 * PAD
        for text, color in lines:
            if text:
                self._emit_text(verts, text, tx, ty, color, sw, sh)
            ty += gh

    # -- public ----------------------------------------------------------
    def build(self, lines, sw, sh):
        """HUD only. Returns (bytes, vertex_count)."""
        verts: list[float] = []
        self._emit_hud(verts, lines, sw, sh)
        arr = np.array(verts, dtype=np.float32)
        return arr.tobytes(), len(verts) // 8

    def build_frame(self, lines, buttons, sw, sh, overlay=None):
        """HUD + control panel (+ optional modal overlay) in one vertex stream.

        Returns (bytes, vertex_count).
        """
        verts: list[float] = []
        self._emit_hud(verts, lines, sw, sh)
        self._emit_panel(verts, buttons, sw, sh)
        if overlay:
            self._emit_overlay(verts, overlay, sw, sh)
        arr = np.array(verts, dtype=np.float32)
        return arr.tobytes(), len(verts) // 8


def compose_lines(sim, camera):
    """Build the objectives + status text model from live simulation state."""
    train = sim.train
    objs = sim.objectives
    lines = [
        ("OPENSUBWAY  \\ Vulkan Subway Sim", YELLOW),
        (f"OBJECTIVES  {objs.completed}/{objs.total}", CYAN),
    ]
    for o in objs.items:
        mark = "[x]" if o.done else "[ ]"
        lines.append((f"{mark} {o.text}", GREEN if o.done else GRAY))

    lines += [
        ("", WHITE),
        (f"At/next : {train.current_station_name()}", WHITE),
        (f"Heading : {train.next_station_name()}", WHITE),
        (f"Speed   : {train.speed_kmh():5.1f} km/h", WHITE),
        (f"Status  : {train.status_text()}", WHITE),
        (f"Reverser: {train.reverser_label()}   Doors: {'OPEN' if train.doors_open else 'SHUT'}", WHITE),
        (f"View    : {camera.mode.upper()}   (C, or 1/2/3)", CYAN),
        ("", WHITE),
        ("TAB Options | C view | H horn | R reset", GRAY),
    ]
    if getattr(sim, "paused", False):
        lines.append(("-- PAUSED (Space) --", YELLOW))
    if objs.all_done():
        lines.append(("ALL OBJECTIVES COMPLETE!", GREEN))
    return lines


def compose_settings_overlay(settings, camera, sim):
    """Options / help overlay content: live settings + full control list."""
    vol = "MUTED" if settings.muted else f"{int(round(settings.volume * 100)):3d}%"
    title = [
        ("= = =  OPTIONS  = = =", YELLOW),
        ("", WHITE),
    ]
    body = [
        ("SETTINGS", CYAN),
        (f"  Camera view ...... {camera.mode.upper()}", WHITE),
        (f"  Paused ........... {'YES' if getattr(sim, 'paused', False) else 'no'}", WHITE),
        (f"  Master volume .... {vol}", WHITE),
        (f"  Field of view .... {int(round(settings.fov))} deg", WHITE),
        (f"  Status HUD ....... {'shown' if settings.hud_visible else 'hidden'}", WHITE),
        ("", WHITE),
        ("CAMERA  (switch the view)", CYAN),
        ("  C ......... cycle drive -> chase -> orbit", GREEN),
        ("  1 / 2 / 3 . drive / chase / orbit directly", GREEN),
        ("  drag+wheel  orbit & zoom (orbit view)", GRAY),
        ("", WHITE),
        ("CONTROLS", CYAN),
        ("  Space ..... pause / resume", WHITE),
        ("  M ......... mute / unmute audio", WHITE),
        ("  - / = ..... master volume down / up", WHITE),
        ("  [ / ] ..... field of view narrow / wide", WHITE),
        ("  F1 ........ show / hide status HUD", WHITE),
        ("  H ......... horn     R ... reset", WHITE),
        ("  Esc ....... quit", WHITE),
        ("", WHITE),
        ("TAB or Esc-menu to close", GRAY),
    ]
    return title + body
