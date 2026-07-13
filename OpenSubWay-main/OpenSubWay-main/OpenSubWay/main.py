"""OpenSubWay entry point.  Run:  python main.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHADER_DIR = ROOT / "opensubway" / "vk" / "shaders"
LAVAPIPE_ICD = ROOT / ".vulkan" / "mesa" / "x64" / "lvp_icd.x86_64.json"


def _ensure_software_vulkan():
    """Use the bundled Mesa lavapipe (CPU) Vulkan driver when present.

    This machine (and many VMs) has no hardware Vulkan driver. If a bundled
    lavapipe ICD is available and the user hasn't opted into a GPU driver via
    OPENSUBWAY_USE_GPU=1, point the Vulkan loader at it. Must run before the
    Vulkan loader initialises (i.e. before importing the app).
    """
    if os.environ.get("OPENSUBWAY_USE_GPU"):
        return
    if LAVAPIPE_ICD.exists() and "VK_ICD_FILENAMES" not in os.environ:
        os.environ["VK_ICD_FILENAMES"] = str(LAVAPIPE_ICD)
        os.environ["VK_DRIVER_FILES"] = str(LAVAPIPE_ICD)
        print(f"Using software Vulkan (lavapipe): {LAVAPIPE_ICD}")


def _ensure_shaders():
    """Compile shaders when SPIR-V is missing or older than its GLSL source."""
    sources = list(SHADER_DIR.glob("*.vert")) + list(SHADER_DIR.glob("*.frag"))
    stale = []
    for source in sources:
        binary = source.with_suffix(source.suffix + ".spv")
        if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
            stale.append(source)
    if stale:
        print("Compiling shaders...")
        sys.path.insert(0, str(ROOT / "tools"))
        import compile_shaders

        if compile_shaders.compile_all() != 0:
            raise SystemExit("Shader compilation failed. Install the Vulkan SDK (glslc).")


def main():
    _ensure_software_vulkan()
    _ensure_shaders()
    from opensubway.app import App

    App().run()


if __name__ == "__main__":
    main()
