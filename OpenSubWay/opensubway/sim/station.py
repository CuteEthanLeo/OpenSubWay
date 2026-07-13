"""Station data model (pure simulation, no rendering)."""

from __future__ import annotations

from dataclasses import dataclass

import glm


@dataclass
class Station:
    name: str
    position: glm.vec3      # centre on the ground (y = 0)
    color: tuple            # (r, g, b) accent
    distance: float = 0.0   # arc-distance along the line (filled in by Line)
