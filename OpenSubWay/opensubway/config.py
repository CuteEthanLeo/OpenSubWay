"""Central configuration: window, colors, timings, and the subway line layout."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Window / rendering
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "OpenSubWay — Vulkan Subway Simulator"

# Background clear color (linear RGBA). The procedural sky pass paints over
# every uncovered pixel, so this only shows for a frame during resizes.
CLEAR_COLOR = (0.66, 0.78, 0.92, 1.0)
SKY_COLOR = (0.66, 0.78, 0.92)   # horizon tint; keep in sync with the shaders

MAX_FRAMES_IN_FLIGHT = 2

# Anti-aliasing: desired MSAA sample count (clamped to what the device supports).
MSAA_SAMPLES = 4

# Enable Vulkan validation layers (developer builds). Set False to silence.
ENABLE_VALIDATION = True

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
FOV_DEGREES = 60.0
NEAR_PLANE = 0.1
FAR_PLANE = 1000.0

# ---------------------------------------------------------------------------
# Simulation timings (seconds / world-units)
# ---------------------------------------------------------------------------
TRAIN_MAX_SPEED = 22.0     # top speed (units/sec)
TRAIN_ACCEL = 7.0          # full-throttle acceleration (units/sec^2)
SIM_TIMESTEP = 1.0 / 120.0  # fixed simulation step

# --- Manual driving (player controls the train from the panel) ---
THROTTLE_NOTCHES = 4       # power handle positions (0..N)
BRAKE_NOTCHES = 4          # brake handle positions (0..N)
BRAKE_DECEL = 10.0         # deceleration at full brake (units/sec^2)
ROLLING_RESISTANCE = 1.2   # passive deceleration when coasting (units/sec^2)
STOP_EPSILON = 0.25        # |speed| below this counts as "stopped" (units/sec)
PLATFORM_STOP_ZONE = 7.0   # must stop within this distance of a station node

# --- Passengers (reactive NPCs) ---
PASSENGER_MAX_PER_STATION = 6
PASSENGER_SPAWN_INTERVAL = 4.0   # seconds between new arrivals at a platform
PASSENGER_BOARD_TIME = 0.8       # seconds for one passenger to board

# --- Procedural city ---
CITY_BLOCK = 22.0          # grid spacing of city blocks (units)
CITY_RADIUS = 260.0        # city extends this far from the line centroid
CITY_TRACK_CLEARANCE = 16.0  # keep buildings this far from the rails
CITY_SEED = 1234

# ---------------------------------------------------------------------------
# Subway line — a closed loop of stations on the XZ ground plane.
# Each entry: (name, x, z, (r, g, b) accent color)
# ---------------------------------------------------------------------------
STATIONS = [
    ("Central",   0.0,   0.0,  (0.90, 0.30, 0.30)),
    ("Riverside", 60.0, -20.0, (0.30, 0.60, 0.90)),
    ("Market",    95.0,  30.0, (0.95, 0.75, 0.25)),
    ("Uptown",    70.0,  85.0, (0.45, 0.80, 0.45)),
    ("Parkway",   10.0, 100.0, (0.70, 0.45, 0.85)),
    ("Depot",    -40.0,  45.0, (0.85, 0.55, 0.35)),
]

# Visual dimensions
RAIL_HALF_WIDTH = 1.2      # half the gauge between the two rails
STATION_SIZE = (10.0, 7.0, 8.0)   # station building (x, y, z)
TRAIN_SIZE = (6.0, 3.0, 3.0)      # single car (length, height, width)

# Daylight palette (albedo values; the shader adds sun + sky lighting)
GROUND_COLOR = (0.31, 0.41, 0.19)    # summer grass (sRGB; shader linearizes)
RAIL_COLOR = (0.55, 0.55, 0.58)      # polished steel
TRAIN_COLOR = (0.88, 0.89, 0.90)     # white EMU body (green stripe added in mesh)
ROAD_COLOR = (0.30, 0.30, 0.33)
