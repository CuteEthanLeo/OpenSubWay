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
FAR_PLANE = 100000.0

# ---------------------------------------------------------------------------
# Simulation timings (seconds / world-units)
# ---------------------------------------------------------------------------
TRAIN_MAX_SPEED = 44.0     # 158.4 km/h, matching the 160 km/h airport line
TRAIN_ACCEL = 5.0          # full-throttle acceleration (units/sec^2)
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
CITY_RADIUS = 1200.0       # local scenery corridor width; city generation follows the line
CITY_TRACK_CLEARANCE = 16.0  # keep buildings this far from the rails
CITY_SEED = 1234

# Railway traffic / infrastructure
OPPOSING_TRACK_OFFSET = -6.4   # second running line, measured along route normal
OPPOSING_TRAIN_SPEED = 25.0    # ~90 km/h average, giving the public 39 min run
AIRPORT_HEADWAY_SECONDS = 14.0 * 60.0
OPPOSING_SERVICE_COUNT = 3
SIGNAL_SPACING = 950.0     # roughly one block signal per kilometre
SIGNAL_BLOCK_LENGTH = 1050.0

# ---------------------------------------------------------------------------
# Subway line — a closed loop of stations on the XZ ground plane.
# Each entry: (name, x, z, (r, g, b) accent color)
# ---------------------------------------------------------------------------
STATIONS = [
    ("虹桥2号航站楼站",             0.0,   0.0,    0.0, (0.10, 0.45, 0.82), False),
    ("中春路站",                  5200.0,   0.0,  -80.0, (0.10, 0.52, 0.84), False),
    ("景洪路站",                 19892.0, -14.0,  220.0, (0.12, 0.50, 0.86), True),
    ("三林南站",                 24897.0, -14.0,  420.0, (0.15, 0.48, 0.84), True),
    ("康桥东站",                 37400.0, -14.0,  180.0, (0.12, 0.46, 0.82), True),
    ("上海国际旅游度假区站",     43465.0, -14.0,  520.0, (0.14, 0.48, 0.86), True),
    ("浦东1号2号航站楼站",       58578.0, -14.0,  260.0, (0.08, 0.42, 0.80), True),
]
LINE_CLOSED = False
TERMINAL_TRACK_EXTENSION = 260.0
REAL_ROUTE_LENGTH_KM = 58.578

# Catmull-Rom samples per station-to-station span.  More samples produce
# genuinely curved rails and continuous train yaw instead of corner snapping.
TRACK_CURVE_SAMPLES = 24

# Secondary alignments are scenery/yard routes connected to the main line by
# visible turnout geometry.  The player service runs around the expanded main
# loop; these establish room for a future route selector without faking extra
# city scale.
BRANCH_LINES = [
    [(-80.0, 2.0), (-260.0, -55.0), (-520.0, -120.0), (-820.0, -105.0)],
    [(4850.0, -72.0), (5050.0, -260.0), (5350.0, -420.0), (5700.0, -455.0)],
    [(5010.0, -248.0), (5250.0, -520.0), (5580.0, -610.0)],
    [(5320.0, -74.0), (5580.0, 125.0), (5900.0, 245.0)],
]

# Visual dimensions
RAIL_HALF_WIDTH = 0.7175   # half of the standard 1,435 mm rail gauge
STATION_SIZE = (205.0, 8.5, 13.0)  # normal 8-car-capable platform / hall
LONG_PLATFORM_LENGTH = 405.0       # terminals and Resort reserve 16-car length
TRAIN_SIZE = (24.5, 3.8, 3.3)      # actual Class C express car dimensions
TRAIN_CARS = 4
TRAIN_CAR_GAP = 1.0

# Daylight palette (albedo values; the shader adds sun + sky lighting)
GROUND_COLOR = (0.31, 0.41, 0.19)    # summer grass (sRGB; shader linearizes)
RAIL_COLOR = (0.66, 0.67, 0.70)      # bright polished steel rail head
TRAIN_COLOR = (0.91, 0.92, 0.94)     # white Airport Link Line EMU body
ROAD_COLOR = (0.30, 0.30, 0.33)
