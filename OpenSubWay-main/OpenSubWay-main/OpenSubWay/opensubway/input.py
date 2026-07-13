"""GLFW input handling: orbit camera, zoom, and control keys."""

from __future__ import annotations

import glfw


class InputState:
    def __init__(self, window, camera, renderer, app, panel):
        self.window = window
        self.camera = camera
        self.renderer = renderer
        self.app = app
        self.panel = panel
        self._last = None  # last cursor pos while dragging

        glfw.set_framebuffer_size_callback(window, self._on_resize)
        glfw.set_mouse_button_callback(window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(window, self._on_cursor)
        glfw.set_scroll_callback(window, self._on_scroll)
        glfw.set_key_callback(window, self._on_key)

    def _on_resize(self, window, width, height):
        self.renderer.framebuffer_resized = True

    def _on_mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            fbw, fbh = glfw.get_framebuffer_size(window)
            ww, wh = glfw.get_window_size(window)
            # Cursor is in window coords; the HUD/panel are in framebuffer pixels.
            px = x * (fbw / ww if ww else 1.0)
            py = y * (fbh / wh if wh else 1.0)
            hit = self.panel.hit_test(px, py, fbw, fbh)
            if hit:
                self.panel.apply(hit)
                return                    # consumed by the panel
            self._last = (x, y)           # otherwise begin an orbit drag
        elif action == glfw.RELEASE:
            self._last = None

    def _on_cursor(self, window, xpos, ypos):
        if self._last is None:
            return
        dx = xpos - self._last[0]
        dy = ypos - self._last[1]
        self._last = (xpos, ypos)
        self.camera.orbit(-dx * 0.01, -dy * 0.01)

    def _on_scroll(self, window, xoffset, yoffset):
        self.camera.zoom(0.9 if yoffset > 0 else 1.1)

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            # Esc closes the Options menu first; otherwise it quits.
            if self.app.settings.show_menu:
                self.app.toggle_settings()
            else:
                glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_TAB:
            self.app.toggle_settings()
        # --- camera ---
        elif key == glfw.KEY_C:
            self.camera.cycle_mode()
        elif key == glfw.KEY_1:
            self.app.set_camera("drive")
        elif key == glfw.KEY_2:
            self.app.set_camera("chase")
        elif key == glfw.KEY_3:
            self.app.set_camera("orbit")
        elif key == glfw.KEY_V:
            self.camera.reset_look()
        # --- settings ---
        elif key == glfw.KEY_SPACE:
            self.app.toggle_pause()
        elif key == glfw.KEY_M:
            self.app.toggle_mute()
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.app.volume_down()
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.app.volume_up()
        elif key == glfw.KEY_LEFT_BRACKET:
            self.app.fov_down()
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.app.fov_up()
        elif key == glfw.KEY_F1:
            self.app.toggle_hud()
        # --- other ---
        elif key == glfw.KEY_H:
            self.app.horn()
        elif key == glfw.KEY_R:
            self.app.reset()
