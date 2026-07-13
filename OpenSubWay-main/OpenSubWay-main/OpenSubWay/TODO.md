# OpenSubWay — Development Roadmap

## Done

- [x] **M1 — Scaffold + shader toolchain.** Project layout, deps, `glslc`
      integration, and resolving software Vulkan (Mesa lavapipe) for this
      GPU-less VM.
- [x] **M2 — Vulkan window + spinning cube.** Instance/device/surface, swapchain
      + depth, scene pipeline, per-frame command recording, present. Confirmed
      rendering works on lavapipe. Zero validation errors.
- [x] **M3 — World geometry + camera.** Ground, looped track bed + rails,
      station buildings with colored roofs, train car. Orbit camera.
- [x] **M4 — Train state machine.** Accelerate / cruise / decelerate / dwell,
      exact stops at station nodes, looped line, current/next station tracking.
      Objectives model. Headless pytest (10 tests).
- [x] **M5 — Text HUD.** Pillow font atlas → R8 texture, alpha-blended text
      pipeline with a descriptor set, per-frame dynamic vertex buffers.
      Objectives panel + live status readout.
- [x] **First-person drive camera** (default), plus chase and orbit; cycle `C`.

## Polish (M6)

- [x] Reset / camera-cycle controls.
- [x] Window-resize handling (dynamic viewport + swapchain recreation).
- [x] README with run instructions and controls.
- [x] `docs/screenshot.png` for the README.

## V2 — "make it real" upgrade (complete)

- [x] **Manual driving**: player-controlled throttle/brake/reverser/doors
      (replaced the auto state machine). Headless driving tests.
- [x] **4× MSAA** anti-aliasing (multisampled color+depth, resolve pass).
- [x] **Night city**: procedural building grid with emissive lit windows,
      lit platforms + lamp posts, night lighting + distance fog.
- [x] **Reactive passengers** (NPCs): wait on platforms, board on doors-open.
- [x] **Sound**: numpy-synthesized rumble (speed-driven), chime, horn, ambience
      via pygame.mixer; graceful with no device.
- [x] **Clickable control panel**: throttle/brake/reverser/doors/horn, mouse
      hit-testing; panel test.

## V3 — realistic daylight graphics (complete)

- [x] **Procedural sky pass**: fullscreen triangle at far depth — gradient,
      sun disc + halo, clouds.
- [x] **Daylight lighting rig** in the scene shader: directional sun,
      sky-hemisphere ambient with ground bounce, Blinn-Phong specular, fresnel
      sky sheen, ACES tonemapping + gamma 2.2, aerial-perspective fog that
      warms toward the sun.
- [x] **Daytime art pass**: grass ground + verge, ballast bed, steel rails,
      catenary masts + contact wire, line-side trees, white EMU livery with
      green stripe / window band / headlights, day city facades with glass
      windows.

## V4 — golden-hour "AAA look" pass (complete)

- [x] **Material system**: vertex format extended with (gloss, translucency);
      gloss drives specular tightness/strength + fresnel reflection (rails,
      train paint, glass); translucency lets the low sun bleed through tree
      canopies (backlight + rim).
- [x] **sRGB-correct shading**: vertex colors linearized before lighting
      (gamma-2 approx) — fixes the washed-out pastel "cartoon" look.
- [x] **Procedural surface grain**: dominant-axis planar noise breaks up flat
      albedo (grass patchiness, masonry, ballast speckle), with distance LOD.
- [x] **Golden-hour rig**: warm low sun, amber-to-cool horizon by sun azimuth,
      contact-occlusion ambient, warm aerial perspective.
- [x] **Volumetric-look clouds**: 4-octave FBM cumulus, sun-offset density
      resample for lit faces vs shadowed bases, silver linings, slow drift
      (time via camPos.w).

## V5 — settings + camera UX (complete)

- [x] **In-game Options menu**: centered modal overlay (`Tab`) that dims the
      scene and lists every control plus live settings, drawn through the text
      pipeline. `Esc` closes the menu before it quits the game.
- [x] **Runtime settings** (`GameSettings`): pause (`Space`), audio mute (`M`)
      and master volume (`-`/`=`) wired into `SoundManager`, field of view
      (`[`/`]`), and status-HUD visibility (`F1`).
- [x] **Camera switching UX**: direct `1`/`2`/`3` selection of drive/chase/orbit
      alongside the `C` cycle; the current view + controls are surfaced in the
      HUD and the Options menu. Headless settings tests.

## Backlog / ideas

- [ ] Multi-car trains; visible passengers *inside* the car after boarding.
- [ ] Station name labels floating in 3D (billboard text using the atlas).
- [x] Smoother track: arc-length sampled Hermite rails and interpolated
      train/camera heading across the full 58.6 km Airport Link Line.
- [ ] Perf: instanced draws for the city/people; frustum culling; optional lower
      MSAA / building density to lift FPS on the CPU driver.
- [ ] Walk-cycle animation for boarding passengers.
- [ ] Device-local vertex buffers via staging (currently host-visible).
- [ ] Switch `import glm` → `from pyglm import glm` (pending deprecation).
- [ ] Time-of-day cycle (day → dusk → night) reusing the emissive channel.
- [ ] Sun shadows (shadow map) — likely too costly on the CPU driver.
