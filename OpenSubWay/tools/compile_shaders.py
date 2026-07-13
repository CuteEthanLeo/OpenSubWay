"""Compile GLSL shaders to SPIR-V.

Finds a shader compiler (glslc from the Vulkan SDK, or glslangValidator) and
compiles every ``*.vert`` / ``*.frag`` under ``opensubway/vk/shaders`` to a
matching ``*.spv`` file next to it.

Run:  python tools/compile_shaders.py
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = ROOT / "opensubway" / "vk" / "shaders"


def find_glslc() -> str | None:
    """Locate glslc.exe from VULKAN_SDK, common install dirs, or PATH."""
    # 1. VULKAN_SDK env var
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = Path(sdk) / "Bin" / ("glslc.exe" if os.name == "nt" else "glslc")
        if cand.exists():
            return str(cand)
    # 2. Default Windows install location: C:\VulkanSDK\<version>\Bin\glslc.exe
    if os.name == "nt":
        matches = sorted(glob.glob(r"C:\VulkanSDK\*\Bin\glslc.exe"), reverse=True)
        if matches:
            return matches[0]
    # 3. On PATH
    found = shutil.which("glslc")
    if found:
        return found
    return None


def find_glslang() -> str | None:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        exe = "glslangValidator.exe" if os.name == "nt" else "glslangValidator"
        cand = Path(sdk) / "Bin" / exe
        if cand.exists():
            return str(cand)
    if os.name == "nt":
        matches = sorted(
            glob.glob(r"C:\VulkanSDK\*\Bin\glslangValidator.exe"), reverse=True
        )
        if matches:
            return matches[0]
    return shutil.which("glslangValidator")


def compile_all() -> int:
    sources = sorted(SHADER_DIR.glob("*.vert")) + sorted(SHADER_DIR.glob("*.frag"))
    if not sources:
        print(f"No shader sources found in {SHADER_DIR}")
        return 0

    glslc = find_glslc()
    glslang = None if glslc else find_glslang()
    if not glslc and not glslang:
        print(
            "ERROR: No shader compiler found.\n"
            "  Install the Vulkan SDK (provides glslc / glslangValidator),\n"
            "  or put one on PATH, or set the VULKAN_SDK environment variable."
        )
        return 1

    tool = glslc or glslang
    print(f"Using shader compiler: {tool}")

    failures = 0
    for src in sources:
        out = src.with_suffix(src.suffix + ".spv")
        if glslc:
            cmd = [glslc, str(src), "-o", str(out)]
        else:
            cmd = [glslang, "-V", str(src), "-o", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failures += 1
            print(f"FAILED  {src.name}\n{proc.stdout}\n{proc.stderr}")
        else:
            print(f"OK      {src.name} -> {out.name}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(compile_all())
