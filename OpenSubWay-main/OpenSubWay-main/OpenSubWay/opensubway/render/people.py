"""Low-poly person mesh for NPC passengers."""

from __future__ import annotations

from . import mesh
from .mesh import Mesh


def build_person_mesh(coat=(0.90, 0.55, 0.55)) -> Mesh:
    """A small stylised person (~1.8 units tall), origin at the feet.

    Lit by the daylight rig like everything else (no self-illumination).
    """
    parts = [
        # legs
        mesh.make_box(size=(0.42, 0.9, 0.32), color=(0.28, 0.28, 0.36),
                      center=(0.0, 0.45, 0.0)),
        # torso / coat
        mesh.make_box(size=(0.56, 0.9, 0.36), color=coat, center=(0.0, 1.2, 0.0)),
        # head
        mesh.make_box(size=(0.34, 0.34, 0.34), color=(0.96, 0.82, 0.70),
                      center=(0.0, 1.78, 0.0)),
    ]
    return mesh.combine(parts)
