import math, datetime, base64
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, no_update, html,dcc
import base64, pathlib
from dash.exceptions import PreventUpdate
from LogReportGenerator import generate_report_bundle
from multi_region_addon import mount_multi_region, multi_hidden_components


# ========================================================================================================================
# ==============================================   TARGETED FLIGHT   =====================================================
#
#  Functions for visualizing and reporting targeted drone flights inside voxelized room models. 
#  These utilities allow users to focus on specific regions of interest, preview drone orbits, 
#  and generate detailed reports with embedded 3D figures and summaries.
#
#  Features:
#   - render voxel grids and room boundaries in 3D
#   - highlight selected voxels and their bounding boxes
#   - generate orbit flight paths around a target cell center
#   - preview both overview and zoomed flight plans interactively
#   - export reports with markdown summaries and inline visualizations
# ========================================================================================================================



# -----------------------------
# data structures 
# -----------------------------
@dataclass
class DroneConfig:
    weight: float
    max_battery_time: float
    max_distance: float
    horizontal_fov: float
    vertical_fov: float
    fps: float
    resolution: Tuple[int, int]
    speed: float
    min_altitude: float
    hover_buffer: float
    battery_warning_threshold: float

@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    gimbal_pitch: float
    speed: float
    hold_time: float

# -----------------------------
# granularity helpers 
# -----------------------------
def choose_equal_counts(dims: Tuple[float,float,float], target_total_cubes: int) -> Tuple[int,int,int]:
    """
    choose integer (nx, ny, nz) such that nx*ny*nz≈target_total_cubes and the cell size is uniform on each axis.
    the room is split evenly with dx=W/nx, dy=L/ny, dz=H/nz.

    args:
        dims: (W, L, H) room dimensions in meters.
        target_total_cubes: desired approximate number of voxels.
    returns:
        (nx, ny, nz): counts per axis that tile the room with equal-sized cells.
    """
    W, L, H = dims
    target_total_cubes = max(1, int(target_total_cubes))  # guard
    V = max(1e-12, W*L*H)

    e0 = (V / target_total_cubes) ** (1.0/3.0)

    #initial integer counts (rounded) and ensure >=1
    nx0 = max(1, int(round(W / e0)))
    ny0 = max(1, int(round(L / e0)))
    nz0 = max(1, int(round(H / e0)))

    def cost(nx: int, ny: int, nz: int) -> Tuple[float, float]:
        total = nx * ny * nz
        #uniform cell sizes with these counts
        dx, dy, dz = W / nx, L / ny, H / nz
        aspect = max(dx, dy, dz) / max(1e-12, min(dx, dy, dz))
        return (abs(total - target_total_cubes), aspect)

    best = (nx0, ny0, nz0)
    best_cost = (float("inf"), float("inf"))  

    #explore a small neighborhood around the initial guess
    for dnx in (-1, 0, 1, 2):
        for dny in (-1, 0, 1, 2):
            for dnz in (-1, 0, 1, 2):
                nx = max(1, nx0 + dnx)
                ny = max(1, ny0 + dny)
                nz = max(1, nz0 + dnz)
                c = cost(nx, ny, nz)
                if c < best_cost:
                    best_cost = c
                    best = (nx, ny, nz)

    return best
# -----------------------------
# voxel grid utilities 
# -----------------------------
class VoxelGrid:
    """
    uniform voxelization that exactly tiles the room using equal subdivisions.
    mapping is stable and 1-based:
        id = 1 + i + j*nx + k*nx*ny

    attributes:
        dims: (W, L, H) in meters
        origin: world origin of the grid (x0, y0, z0)
        nx, ny, nz: cell counts per axis
        x_edges, y_edges, z_edges: numpy edges per axis
        dx, dy, dz: cell sizes
        centers_xyz: (N,3) array of voxel centers in world coords
        indices_ijk: list of (i, j, k) per voxel id-1
        ids: numpy array of stable 1-based ids
    """
    def __init__(self, dims: Tuple[float, float, float], nx: int, ny: int, nz: int, origin=(0.0, 0.0, 0.0)):
        self.dims = dims
        self.origin = origin
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)

        W, L, H = dims
        ox, oy, oz = origin

        #equal subdivisions -> identical cell sizes on each axis
        self.x_edges = np.linspace(0.0, W, self.nx + 1)
        self.y_edges = np.linspace(0.0, L, self.ny + 1)
        self.z_edges = np.linspace(0.0, H, self.nz + 1)

        self.dx = float(W / self.nx)
        self.dy = float(L / self.ny)
        self.dz = float(H / self.nz)

        #compute centers, indices, ids
        centers = []
        indices = []
        ids = []
        id_counter = 1
        for k in range(self.nz):
            zc = (self.z_edges[k] + self.z_edges[k+1]) * 0.5
            for j in range(self.ny):
                yc = (self.y_edges[j] + self.y_edges[j+1]) * 0.5
                for i in range(self.nx):
                    xc = (self.x_edges[i] + self.x_edges[i+1]) * 0.5
                    centers.append((ox + xc, oy + yc, oz + zc))
                    indices.append((i, j, k))
                    ids.append(id_counter)
                    id_counter += 1

        self.centers_xyz = np.array(centers)
        self.indices_ijk = indices
        self.ids = np.array(ids, dtype=int)

    def ijk_to_id(self, i: int, j: int, k: int) -> int:
        """convert 0-based (i,j,k) to stable 1-based voxel id."""
        return 1 + i + j * self.nx + k * self.nx * self.ny

    def id_to_ijk(self, id1: int) -> Tuple[int, int, int]:
        """convert stable 1-based voxel id to 0-based (i,j,k)."""
        id0 = id1 - 1
        k = id0 // (self.nx * self.ny)
        r = id0 % (self.nx * self.ny)
        j = r // self.nx
        i = r % self.nx
        return (i, j, k)

    def index_to_center(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """world-space center of the voxel at (i,j,k)."""
        ox, oy, oz = self.origin
        xc = (self.x_edges[i] + self.x_edges[i+1]) * 0.5
        yc = (self.y_edges[j] + self.y_edges[j+1]) * 0.5
        zc = (self.z_edges[k] + self.z_edges[k+1]) * 0.5
        return (ox + xc, oy + yc, oz + zc)

    def index_to_bounds(self, i: int, j: int, k: int) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
        """axis-aligned bounds ((xmin,ymin,zmin),(xmax,ymax,zmax)) for voxel (i,j,k)."""
        ox, oy, oz = self.origin
        return ((ox + self.x_edges[i],   oy + self.y_edges[j],   oz + self.z_edges[k]),
                (ox + self.x_edges[i+1], oy + self.y_edges[j+1], oz + self.z_edges[k+1]))

# -----------------------------
# orbit flight-plan generator 
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

###### previous 
def generate_cube_orbit(
    drone: DroneConfig,
    dims: Tuple[float, float, float],
    center: Tuple[float, float, float],
    radius: float = 0.6,
    n_views: int = 12,
    dwell_s: float = 2.0,
    wall_clearance: float = 0.05,
    floor_clearance: float = 0.0,
    ceiling_clearance: float = 0.0
) -> List[Waypoint]:
    """
    generate a horizontal ring of viewpoints around a voxel center at a safe z.

    behavior:
        - prunes viewpoints outside room bounds (with wall_clearance margin)
        - if too few samples remain, shrinks radius (×0.7) and retries
        - respects floor/ceiling clearances when choosing z
        - returns waypoints ordered by polar angle for smooth pathing

    args:
        drone: DroneConfig with speed, etc.
        dims: (W, L, H) room dimensions in meters.
        center: (cx, cy, cz) target center in meters.
        radius: ring radius in meters.
        n_views: number of viewpoints around the ring.
        dwell_s: hover time at each waypoint.
        wall_clearance: min distance to walls along x/y.
        floor_clearance: min z above floor.
        ceiling_clearance: min distance below ceiling.
    returns:
        list of Waypoint instances forming a closed loop (last->first adjacent).
    """
    w, l, h = dims
    cx, cy, cz = center

    low  = max(wall_clearance, floor_clearance)
    high = max(low, h - max(wall_clearance, ceiling_clearance))
    z = (h * 0.5) if (high <= low) else clamp(cz, low, high)

    def ring(R: float) -> List[Waypoint]:
        wps = []
        for a in np.linspace(0.0, 2*np.pi, num=n_views, endpoint=False):
            x = cx + R * np.cos(a)
            y = cy + R * np.sin(a)
            if (wall_clearance <= x <= w - wall_clearance) and (wall_clearance <= y <= l - wall_clearance):
                wps.append(Waypoint(x=float(x), y=float(y), z=float(z),
                                    gimbal_pitch=0.0, speed=drone.speed, hold_time=dwell_s))
        return wps

    wps = ring(radius)
    if len(wps) < 3:
        wps = ring(radius * 0.7)

    def angle(p): 
        return math.atan2(p.y - cy, p.x - cx)
    wps.sort(key=angle)
    if wps:
        idx0 = min(range(len(wps)), key=lambda i: abs(angle(wps[i]) - 0.0))
        wps = wps[idx0:] + wps[:idx0]

    return wps

######end previous 







####### start 
#--- waypoint builder ---
def _make_wp(x, y, z, heading_deg, gimbal_deg, speed, hold) -> Waypoint:
    """
    create a Waypoint and attach heading (in degrees) as an extra attribute.
    this keeps your dataclass unchanged while letting downstream code read wp.heading_deg.
    """
    wp = Waypoint(
        x=float(x),
        y=float(y),
        z=float(z),
        gimbal_pitch=float(gimbal_deg),
        speed=float(speed),
        hold_time=float(hold or 0.0),
    )
    # attach constant-heading info without modifying the dataclass
    setattr(wp, "heading_deg", float(heading_deg))
    return wp


#--- clamp position inside box with safety margin ---
def _clamp_box(x: float, y: float, z: float,
               dims: Tuple[float,float,float], margin: float) -> Tuple[float,float,float]:
    """clamp (x,y,z) so it stays within [margin, dim-margin]"""
    W, L, H = dims
    x = min(max(x, margin), W - margin)
    y = min(max(y, margin), L - margin)
    z = min(max(z, margin), H - margin)
    return (x, y, z)


#--- convert heading degrees to unit vector in xy plane ---
def _heading_vec_from_deg(heading_deg: float) -> Tuple[float,float,float]:
    """return unit vector (x,y,0) for given heading (0°=+x, 90°=+y)"""
    rad = math.radians(heading_deg)
    return (math.cos(rad), math.sin(rad), 0.0)


#--- orthogonal xy vector to a given (hx,hy) ---
def _orth_xy(hx: float, hy: float) -> Tuple[float,float]:
    """return perpendicular unit vector in xy plane"""
    return (-hy, hx)

#--- auto heading from nearest wall (face away from the closest wall) ---
def _auto_heading_from_nearest_wall(dims: Tuple[float,float,float],
                                    point: Tuple[float,float,float]) -> float:
    """
    return a heading (deg) that points away from the nearest room wall in XY.
    walls: x=0 (−x), x=W (+x), y=0 (−y), y=L (+y)
    """
    W, L, _ = dims
    cx, cy, _ = point

    # distances to each wall
    d = {
        "-x": cx,
        "+x": W - cx,
        "-y": cy,
        "+y": L - cy,
    }
    nearest = min(d, key=d.get)

    # choose heading that points *away* from that wall
    if nearest == "-x":  # closest to x=0 → face +x
        return 0.0
    if nearest == "+x":  # closest to x=W → face -x
        return 180.0
    if nearest == "-y":  # closest to y=0 → face +y
        return 90.0
    # nearest == "+y": closest to y=L → face -y
    return 270.0


#--- helpers for pillar+connector sequencing (prevents diagonal jumps) ---

def plan_free_object_perimeter_loop(
    dims: Tuple[float,float,float],
    drone: DroneConfig,
    obj_center: Tuple[float,float,float],
    obj_size_xy: Tuple[float,float],          # (size_x, size_y) of the object footprint
    desired_low_z: float,
    desired_top_z: float,
    climb_gimbal_deg: float = 0.0,            # forward-looking during verticals
    connector_gimbal_deg: float = -60.0,      # down-looking during horizontals
    safety_margin: float = 0.12,              # keep inside the room
) -> List[Waypoint]:
    """
    build a 3D-lawnmower perimeter loop with exactly three vertical levels:
      bottom → mid → top at corner 0, then top-level horizontal to corner 1,
      descend at corner 1, bottom-level horizontal to corner 2,
      descend at corner 2, top-level horizontal to corner 3,
      descend at corner 3, bottom-level horizontal back to corner 0.

    notes
    - this produces only horizontal segments between pillars (no diagonals).
    - all points are clamped to the room bounds using 'safety_margin'.
    - returns Waypoints with (x,y,z,gimbal_pitch,speed,hold_time).
    """

    #---unpack room and object inputs---
    W, L, H = dims
    cx, cy, cz = obj_center
    sx, sy = obj_size_xy

    #---compute three discrete altitudes and clamp to room---
    z_low = max(safety_margin, min(H - safety_margin, float(desired_low_z)))
    z_top = max(safety_margin, min(H - safety_margin, float(desired_top_z)))
    if z_top < z_low:
        z_low, z_top = z_top, z_low
    z_mid = 0.5 * (z_low + z_top)

    #---derive the XY corners from center+size (counter-clockwise order)---
    hx = 0.5 * float(sx)
    hy = 0.5 * float(sy)
    c0x, c0y = cx - hx, cy - hy
    c1x, c1y = cx + hx, cy - hy
    c2x, c2y = cx + hx, cy + hy
    c3x, c3y = cx - hx, cy + hy
    corners: List[Tuple[float,float]] = [(c0x,c0y), (c1x,c1y), (c2x,c2y), (c3x,c3y)]

    #--- local helpers ---
    def _clamped_wp(x, y, z, gimbal, hold=0.0):
        xx, yy, zz = _clamp_box(x, y, z, dims, margin=safety_margin)
        return Waypoint(x=xx, y=yy, z=zz, gimbal_pitch=gimbal, speed=drone.speed, hold_time=float(hold))

    def _add_pillar_up(wps: List[Waypoint], x: float, y: float):
        #bottom→mid→top
        wps.append(_clamped_wp(x, y, z_low, climb_gimbal_deg, drone.hover_buffer))  #bottom
        wps.append(_clamped_wp(x, y, z_mid, climb_gimbal_deg, 0.0))                 #mid
        wps.append(_clamped_wp(x, y, z_top, climb_gimbal_deg, 0.0))                 #top

    def _add_pillar_down(wps: List[Waypoint], x: float, y: float):
        #top→mid→bottom
        wps.append(_clamped_wp(x, y, z_top, climb_gimbal_deg, 0.0))                 #top (anchor)
        wps.append(_clamped_wp(x, y, z_mid, climb_gimbal_deg, 0.0))                 #mid
        wps.append(_clamped_wp(x, y, z_low, climb_gimbal_deg, 0.0))                 #bottom

    def _connect_horizontal(wps: List[Waypoint], x_from: float, y_from: float,
                            x_to: float, y_to: float, z_level: float):
        #two explicit points at the same altitude -> straight horizontal edge
        #start is implicit as last waypoint; we add the destination at same z
        wps.append(_clamped_wp(x_to, y_to, z_level, connector_gimbal_deg, 0.0))

    #---build the loop---
    #---build the loop (ABAB pattern: top/bottom connectors alternate)---
    wps: List[Waypoint] = []

    # c0: pillar up (bottom→mid→top)
    x0, y0 = corners[0]
    _add_pillar_up(wps, x0, y0)

    # top connector to c1, then c1: pillar down
    x1, y1 = corners[1]
    _connect_horizontal(wps, x0, y0, x1, y1, z_top)
    _add_pillar_down(wps, x1, y1)

    # bottom connector to c2, then c2: pillar up
    x2, y2 = corners[2]
    _connect_horizontal(wps, x1, y1, x2, y2, z_low)
    _add_pillar_up(wps, x2, y2)

    # top connector to c3, then c3: pillar down
    x3, y3 = corners[3]
    _connect_horizontal(wps, x2, y2, x3, y3, z_top)
    _add_pillar_down(wps, x3, y3)

    # bottom connector back to c0 (closes loop at bottom; no extra pillar)
    _connect_horizontal(wps, x3, y3, x0, y0, z_low)


    return wps

def plan_free_object_lawnmower(
    dims: Tuple[float,float,float],
    drone: DroneConfig,
    obj_center: Tuple[float,float,float],
    heading_deg: Optional[float] = None,   
    n_columns: int = 4,                     #how many vertical lines
    lateral_spacing: float = 0.7,           #meters between columns (perpendicular to heading)
    standoff: float = 1.2,                  #how far behind the object to place the columns (along -heading)
    desired_low_z: float = 0.6,
    desired_top_z: float = 1.8,
    safety_margin: float = 0.15,
    connector_len: float = 0.7,             #short forward hop between columns
    climb_gimbal_deg: float = 0.0,          #forward look on climbs
    connector_gimbal_deg: float = -60.0,    #look down-ish on connectors
) -> List[Waypoint]:
    
    W, L, H = dims
    cx, cy, cz = obj_center

    if heading_deg is None:
        
        d_left, d_right = cx, (W - cx)
        d_bottom, d_top = cy, (L - cy)
        nearest = min((d_left, "left"), (d_right, "right"),
                      (d_bottom, "bottom"), (d_top, "top"), key=lambda t: t[0])[1]
        if nearest == "left":       heading_deg = 0.0    # +x
        elif nearest == "right":    heading_deg = 180.0  # -x
        elif nearest == "bottom":   heading_deg = 90.0   # +y
        else:                       heading_deg = 270.0  # -y

    hx, hy, _ = _heading_vec_from_deg(heading_deg)
    ox, oy   = _orth_xy(hx, hy)  # lateral axis

    #clamp levels to room and compute mid
    z_low = max(0.0, min(H, desired_low_z))
    z_top = max(0.0, min(H, desired_top_z))
    if z_top < z_low: z_low, z_top = z_top, z_low
    z_mid = 0.5*(z_low + z_top)

    #column lateral offsets centered on object
    n_columns = max(1, int(n_columns))
    half = 0.5 * (n_columns - 1)
    lats = [ (i - half) * lateral_spacing for i in range(n_columns) ]

    #column base positions (behind the object along -heading, each at its own lateral offset)
    cols = []
    for lat in lats:
        x = cx - hx * standoff + ox * lat
        y = cy - hy * standoff + oy * lat
        x, y, _ = _clamp_box(x, y, cz, dims, safety_margin)
        cols.append((x, y))

    wps: List[Waypoint] = []

    def pillar_up(x, y):
        xb, yb, zb = _clamp_box(x, y, z_low, dims, safety_margin)
        xm, ym, zm = _clamp_box(x, y, z_mid, dims, safety_margin)
        xt, yt, zt = _clamp_box(x, y, z_top, dims, safety_margin)
        wps.append(_make_wp(xb, yb, zb, heading_deg, climb_gimbal_deg, drone.speed, drone.hover_buffer))
        wps.append(_make_wp(xm, ym, zm, heading_deg, climb_gimbal_deg, drone.speed, 0.0))
        wps.append(_make_wp(xt, yt, zt, heading_deg, climb_gimbal_deg, drone.speed, 0.0))

    def pillar_down(x, y):
        xt, yt, zt = _clamp_box(x, y, z_top, dims, safety_margin)
        xm, ym, zm = _clamp_box(x, y, z_mid, dims, safety_margin)
        xb, yb, zb = _clamp_box(x, y, z_low, dims, safety_margin)
        wps.append(_make_wp(xt, yt, zt, heading_deg, climb_gimbal_deg, drone.speed, 0.0))
        wps.append(_make_wp(xm, ym, zm, heading_deg, climb_gimbal_deg, drone.speed, 0.0))
        wps.append(_make_wp(xb, yb, zb, heading_deg, climb_gimbal_deg, drone.speed, 0.0))

    def forward_hop_at(z_level):
        #short forward move (aligned with heading), keeps object in frame
        x0, y0 = wps[-1].x, wps[-1].y
        xf = x0 + hx * connector_len
        yf = y0 + hy * connector_len
        xf, yf, zf = _clamp_box(xf, yf, z_level, dims, safety_margin)
        wps.append(_make_wp(xf, yf, zf, heading_deg, connector_gimbal_deg, drone.speed, 0.0))

    #ABAB pattern across columns, always using forward hops
    for idx, (x, y) in enumerate(cols):
        if idx == 0:
            pillar_up(x, y)
        else:
            ended_top = (wps[-1].z >= (z_mid + (z_top - z_low)*0.25))
            forward_hop_at(z_top if ended_top else z_low)
            #alternate up/down
            if ended_top:
                pillar_down(x, y)
            else:
                pillar_up(x, y)

    
    forward_hop_at(wps[-1].z)

    return wps

def plan_wall_object_standoffs_3levels(
    dims: Tuple[float,float,float],
    drone: DroneConfig,
    obj_center: Tuple[float,float,float],
    wall_normal_heading_deg: float,
    standoffs: List[float] = [2.0, 1.2, 0.6],  # far→mid→near
    desired_low_z: float = 0.6,
    desired_top_z: float = 1.8,
    safety_margin: float = 0.15,
    connector_len: float = 0.6,
    climb_gimbal_deg: float = 0.0,
    connector_gimbal_deg: float = -75.0,
) -> List[Waypoint]:
    W, L, H = dims
    cx, cy, cz = obj_center
    hx, hy, _ = _heading_vec_from_deg(wall_normal_heading_deg)

    z_low = max(0.0, min(H, desired_low_z))
    z_top = max(0.0, min(H, desired_top_z))
    if z_top < z_low: z_low, z_top = z_top, z_low
    z_mid = 0.5*(z_low + z_top)

    def pillar_up(x, y):
        wps.append(_make_wp(*_clamp_box(x, y, z_low, dims, safety_margin), wall_normal_heading_deg, climb_gimbal_deg, drone.speed, drone.hover_buffer))
        wps.append(_make_wp(*_clamp_box(x, y, z_mid, dims, safety_margin), wall_normal_heading_deg, climb_gimbal_deg, drone.speed, 0.0))
        wps.append(_make_wp(*_clamp_box(x, y, z_top, dims, safety_margin), wall_normal_heading_deg, climb_gimbal_deg, drone.speed, 0.0))

    def pillar_down(x, y):
        wps.append(_make_wp(*_clamp_box(x, y, z_top, dims, safety_margin), wall_normal_heading_deg, climb_gimbal_deg, drone.speed, 0.0))
        wps.append(_make_wp(*_clamp_box(x, y, z_mid, dims, safety_margin), wall_normal_heading_deg, climb_gimbal_deg, drone.speed, 0.0))
        wps.append(_make_wp(*_clamp_box(x, y, z_low, dims, safety_margin), wall_normal_heading_deg, climb_gimbal_deg, drone.speed, 0.0))

    def forward_hop_at(z_level):
        x0, y0 = wps[-1].x, wps[-1].y
        xf = x0 + hx * connector_len
        yf = y0 + hy * connector_len
        xf, yf, zf = _clamp_box(xf, yf, z_level, dims, safety_margin)
        wps.append(_make_wp(xf, yf, zf, wall_normal_heading_deg, connector_gimbal_deg, drone.speed, 0.0))

    wps: List[Waypoint] = []
    for idx, D in enumerate([d for d in standoffs if d is not None]):
        px = cx - hx * D
        py = cy - hy * D
        px, py, _ = _clamp_box(px, py, cz, dims, safety_margin)

        if idx == 0:
            pillar_up(px, py)
        else:
            ended_top = (wps[-1].z >= (z_mid + (z_top - z_low)*0.25))
            forward_hop_at(z_top if ended_top else z_low)
            if ended_top:
                pillar_down(px, py)
            else:
                pillar_up(px, py)

    forward_hop_at(wps[-1].z)
    return wps


def _auto_mode_and_heading(dims, center, wall_margin=0.05):
    W,L,H = dims
    x,y,_ = center
    #near which wall?
    dists = {"left":x, "right":W-x, "bottom":y, "top":L-y}
    wall, d = min(dists.items(), key=lambda kv: kv[1])
    is_wall = d <= wall_margin
    if wall == "left":   head = 0.0
    elif wall == "right":head = 180.0
    elif wall == "bottom":head = 90.0
    else:                head = 270.0
    return ("wall" if is_wall else "free"), head







################ end

# -----------------------------
# plotly helpers 
# -----------------------------
def room_box_edges(dims: Tuple[float,float,float]):
    """return polyline coordinate lists (xs, ys, zs) for the room wireframe box."""
    W, L, H = dims
    corners = np.array([
        [0,0,0],[W,0,0],[W,L,0],[0,L,0],
        [0,0,H],[W,0,H],[W,L,H],[0,L,H],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    xs, ys, zs = [], [], []
    for a,b in edges:
        xs += [corners[a,0], corners[b,0], None]
        ys += [corners[a,1], corners[b,1], None]
        zs += [corners[a,2], corners[b,2], None]
    return xs, ys, zs

def cube_edges_from_bounds(bounds_min, bounds_max):
    """return polyline coordinate lists (xs, ys, zs) for a voxel box from min/max corners."""
    xmin,ymin,zmin = bounds_min
    xmax,ymax,zmax = bounds_max
    corners = np.array([
        [xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin],
        [xmin,ymin,zmax],[xmax,ymin,zmax],[xmax,ymax,zmax],[xmin,ymax,zmax],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    xs, ys, zs = [], [], []
    for a,b in edges:
        xs += [corners[a,0], corners[b,0], None]
        ys += [corners[a,1], corners[b,1], None]
        zs += [corners[a,2], corners[b,2], None]
    return xs, ys, zs

def all_voxel_wireframe_trace(grid: VoxelGrid, line_width: int = 1, name="Voxels"):
    total = grid.nx * grid.ny * grid.nz
    if total > 20000:
        return go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(width=line_width),
            name=f"{name} (omitted for performance)",
            hoverinfo='skip'
        )
    xs, ys, zs = [], [], []
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                bmin, bmax = grid.index_to_bounds(i, j, k)
                ex, ey, ez = cube_edges_from_bounds(bmin, bmax)
                xs += ex; ys += ey; zs += ez
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(width=line_width),
        name=name,
        hoverinfo='skip'
    )



def make_overview_figure(grid: VoxelGrid, show_labels: bool, selected_id: Optional[int]):
    W, L, H = grid.dims
    rx, ry, rz = room_box_edges(grid.dims)
    room_trace = go.Scatter3d(x=rx, y=ry, z=rz, mode='lines', line=dict(width=3), name='Room')

    voxels_trace = all_voxel_wireframe_trace(grid, line_width=1, name="Voxel grid")

    centers = grid.centers_xyz
    ids = grid.ids
    i_vals = [grid.indices_ijk[idx][0] for idx in range(len(ids))]
    j_vals = [grid.indices_ijk[idx][1] for idx in range(len(ids))]
    k_vals = [grid.indices_ijk[idx][2] for idx in range(len(ids))]

    marker_trace = go.Scatter3d(
        x=centers[:,0], y=centers[:,1], z=centers[:,2],
        mode='markers+text' if show_labels else 'markers',
        text=[str(i) for i in ids] if show_labels else None,
        textposition='top center',
        marker=dict(size=2, opacity=0.6),
        name='Voxel centers',
        hovertext=[f"ID {id_}  (i,j,k)=({i},{j},{k})" for id_,i,j,k in zip(ids,i_vals,j_vals,k_vals)],
        hoverinfo='text',
        customdata=np.stack([ids, i_vals, j_vals, k_vals], axis=1)
    )

    highlight_traces = []
    if selected_id is not None and selected_id in ids:
        sel_idx = int(np.where(ids == selected_id)[0][0])
        highlight_traces.append(go.Scatter3d(
            x=[centers[sel_idx,0]], y=[centers[sel_idx,1]], z=[centers[sel_idx,2]],
            mode='markers+text',
            marker=dict(size=6),
            text=[f"ID {selected_id}"],
            textposition='middle right',
            name='Selected center'
        ))
        i,j,k = grid.id_to_ijk(selected_id)
        bmin, bmax = grid.index_to_bounds(i,j,k)
        vx, vy, vz = cube_edges_from_bounds(bmin, bmax)
        highlight_traces.append(go.Scatter3d(
            x=vx, y=vy, z=vz, mode='lines',
            line=dict(width=6),
            name='Selected cell'
        ))

    fig = go.Figure(data=[room_trace, voxels_trace, marker_trace] + highlight_traces)
    fig.update_scenes(
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
        aspectmode='data',
        xaxis=dict(range=[0, W]), yaxis=dict(range=[0, L]), zaxis=dict(range=[0, H])
    )
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig

def make_zoom_figure(
        
    grid: VoxelGrid,
    drone: DroneConfig,
    selected_id: Optional[int],
    radius: float,
    n_views: int,
    dwell: float,
    floor_clearance: float,
    ceiling_clearance: float
):
    """compose the zoomed 3d figure for one selected voxel and its planned circular path."""
    W, L, H = grid.dims
    fig = go.Figure()

    rx, ry, rz = room_box_edges(grid.dims)
    fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode='lines', line=dict(width=2), name='Room'))

    if selected_id is None or selected_id < 1 or selected_id > grid.nx*grid.ny*grid.nz:
        fig.update_layout(
            height=650, margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(xaxis=dict(range=[0,W]), yaxis=dict(range=[0,L]), zaxis=dict(range=[0,H]), aspectmode='data'),
            annotations=[dict(text="Select a cell (click a center or enter ID)", showarrow=False, x=0.5, y=0.5, xref='paper', yref='paper')]
        )
        return fig

    i, j, k = grid.id_to_ijk(selected_id)
    c = grid.index_to_center(i, j, k)
    bmin, bmax = grid.index_to_bounds(i, j, k)
    cx, cy, cz = c

    vx, vy, vz = cube_edges_from_bounds(bmin, bmax)
    fig.add_trace(go.Scatter3d(x=vx, y=vy, z=vz, mode='lines', line=dict(width=5), name=f'Cell ID {selected_id}'))

    xmin,ymin,zmin = bmin
    xmax,ymax,zmax = bmax
    cube_corners = np.array([
        [xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin],
        [xmin,ymin,zmax],[xmax,ymin,zmax],[xmax,ymax,zmax],[xmin,ymax,zmax]
    ])
    i_surf = [0,1,2,3,4,5,6,7,0,1,5,4]
    j_surf = [1,2,3,0,5,6,7,4,4,5,6,7]
    k_surf = [4,5,6,7,0,1,2,3,1,2,6,5]
    fig.add_trace(go.Mesh3d(
        x=cube_corners[:,0], y=cube_corners[:,1], z=cube_corners[:,2],
        i=i_surf, j=j_surf, k=k_surf,
        opacity=0.15, color='lightblue', name='Selected cell'
    ))

    
    levels = max(2, int(n_views or 4))          
    desired_low_z  = 0.6                                 
    desired_top_z  = grid.dims[2] - 0.3                  
    #center of selected cube
    i, j, k = grid.id_to_ijk(selected_id)
    cx, cy, cz = grid.index_to_center(i, j, k)

    #size of the selected voxel footprint (dx, dy)
    dx = grid.dx
    dy = grid.dy

#################################################################################################### orbit
    wps = generate_cube_orbit(
    drone=drone, dims=grid.dims, center=(cx, cy, cz),
    radius=radius, n_views=n_views, dwell_s=dwell,
    wall_clearance=0.02, floor_clearance=floor_clearance, ceiling_clearance=ceiling_clearance
)
#################################################################################################### lawnmower vertical 
#    mode, head = _auto_mode_and_heading(grid.dims, (cx,cy,cz), wall_margin=0.05)
#
#    if mode == "wall":
#        wps = plan_wall_object_standoffs_3levels(
#            dims=grid.dims, drone=DRONE, obj_center=(cx,cy,cz),
#            wall_normal_heading_deg=head,
#            standoffs=[2.0, 1.2, 0.6], desired_low_z=0.6, desired_top_z=1.6
#        )
#    else:
#        wps = plan_free_object_lawnmower(
#            dims=grid.dims, drone=DRONE, obj_center=(cx,cy,cz),
#            heading_deg=head, n_columns=3, lateral_spacing=0.8,
#            standoff=1.2, desired_low_z=0.6, desired_top_z=1.6
#        )
#################################################################################################### lawnmower perimeter
    #wps = plan_free_object_perimeter_loop(
    #    dims=grid.dims,
    #    drone=DRONE,
    #    obj_center=(cx, cy, cz),
    #    obj_size_xy=(dx, dy),         # <- footprint of the selected cube
    #    desired_low_z=0.6,
    #    desired_top_z=1.6,
    #    climb_gimbal_deg=0.0,
    #    connector_gimbal_deg=-60.0,
    #    safety_margin=0.12,
    #)


    if wps:
        xs = [wp.x for wp in wps] + [wps[0].x]
        ys = [wp.y for wp in wps] + [wps[0].y]
        zs = [wp.z for wp in wps] + [wps[0].z]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines+markers', marker=dict(size=4), name='Flight path'))
        fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[zs[0]], mode='markers', marker=dict(size=5, symbol='x'), name='ROI center'))

    fig.update_layout(
        height=650, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(range=[0, W], title='X (m)'),
            yaxis=dict(range=[0, L], title='Y (m)'),
            zaxis=dict(range=[0, H], title='Z (m)'),
            aspectmode='data'
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig


#--------------------------------
# report Generation 
#--------------------------------

def _b64_from_upload(contents: str):
    """dcc.Upload.contents -> base64 string (no header)."""
    if not contents:
        return None
    try:
        _, b64 = contents.split(",", 1)
        return b64
    except Exception:
        return None

def _report_html_from_bundle(bundle: dict, title: str = "Flight Report") -> str:
    """Minimal standalone HTML containing the markdown + all figures inline."""
    report_md = bundle.get("report_md", "")
    sections = bundle.get("sections", [])
    figures  = bundle.get("figures", {})

    fig_blocks = []
    for sec in sections:
        stitle = sec.get("title", "")
        names = sec.get("figs", [])
        imgs = []
        for fname in names:
            b64 = figures.get(fname)
            if not b64:
                continue
            imgs.append(f'<figure><img src="data:image/png;base64,{b64}" alt="{fname}"><figcaption>{fname}</figcaption></figure>')
        if imgs:
            fig_blocks.append(f"<h2>{stitle}</h2>" + "\n".join(imgs))

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body {{ margin:16px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:#111827; }}
 h1,h2 {{ margin:16px 0 8px; }}
 pre {{ background:#f8fafc; padding:12px; border-radius:8px; border:1px solid #e5e7eb; overflow:auto; }}
 figure {{ margin:12px 0; }}
 img {{ max-width:100%; height:auto; border:1px solid #e5e7eb; border-radius:8px; }}
 figcaption {{ font-size:12px; color:#6b7280; margin-top:4px; }}
 .tag {{ display:inline-block; padding:2px 8px; background:#eef2ff; border:1px solid #c7d2fe; border-radius:999px; font-size:12px; color:#3730a3; }}
 .ok  {{ color:#16a34a; font-weight:600; }}
 .bad {{ color:#dc2626; font-weight:600; }}
</style>
</head>
<body>
<h1>Flight Report</h1>
<h2>Summary (Markdown)</h2>
<pre>{report_md}</pre>
{''.join(fig_blocks)}
</body>
</html>"""

def _data_url_from_html(html_text: str) -> str:
    b = html_text.encode("utf-8")
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:text/html;base64,{b64}"

# ----------------------------
# End of report generation 
# ----------------------------





# -----------------------------
# App setup
# -----------------------------
#default room/drone
ROOM_DIMS = (7.2, 3.8, 2.1)    
DEFAULT_FLOOR_CLEAR = 0.0
DEFAULT_CEIL_CLEAR = 0.0
TARGET_LARGER_CELLS  = 24
TARGET_SMALLER_CELLS = 36

#initial grid
_init_nx, _init_ny, _init_nz = choose_equal_counts(ROOM_DIMS, target_total_cubes=TARGET_LARGER_CELLS)
grid = VoxelGrid(ROOM_DIMS, _init_nx, _init_ny, _init_nz)

DRONE = DroneConfig(
    weight=0.292,
    max_battery_time=1380.0,
    max_distance=5000.0,
    horizontal_fov=82.1,
    vertical_fov=52.3,
    fps=60.0,
    resolution=(2720,1530),
    speed=0.4,
    min_altitude=1.0,
    hover_buffer=10.0,
    battery_warning_threshold=0.85
)


def render_marvelmind_csv(waypoints: List[Waypoint]) -> str:
    #index, x, y, z, HoldTime, GimblePitch
    lines = ["index,x,y,z,HoldTime,GimblePitch"]
    for idx, wp in enumerate(waypoints, start=1):
        lines.append(f"{idx},{wp.x:.6f},{wp.y:.6f},{wp.z:.6f},{wp.hold_time:.3f},{wp.gimbal_pitch:.2f}")
    return "\n".join(lines)


def render_dpt(waypoints: List[Waypoint], drone_config: DroneConfig) -> str:
    lines = []
    lines.append(f"takeoff({drone_config.min_altitude:.2f})")
    lines.append("pause(4.0)")
    lines.append(f"height({drone_config.min_altitude:.2f})")
    lines.append("pause(1.0)")
    lines.append("waypoints_begin()")
    for i, wp in enumerate(waypoints):
        lines.append(f"W{i+1:02d}({wp.x:.2f},{wp.y:.2f},{wp.z:.2f})")
    lines.append("waypoints_end()")
    lines.append("pause(1.0)")
    lines.append("landing()")
    lines.append("pause(6.0)")
    lines.append("landing()")
    lines.append("\n[settings]")
    lines.extend([
        "set_power(10.00)",
        "set_power_rot(46.67)",
        "set_power_height(6.67)",
        "set_waypoint_pause(4.0)",
        "set_timeout(9.0)",
        "set_scan_timeout(10.0)",
        "set_waypoint_radius(0.05)",
        "set_wp1_radius_coef(5.0)",
        "set_waypoint_radius_z(0.10)",
        "set_recalibrate_distance(0.50)",
        "set_recalibrate_deviation(0.10)",
        "set_min_rotation_angle(10)",
        "set_angle_control_mode(0)",
        "set_pid_angle_distance(0.30)",
        "set_pid_angle_p(0.050)",
        "set_pid_angle_i(0.005)",
        "set_pid_angle_d(0.005)",
        "set_sliding_window(10)",
        "set_jump_sigma(0.100)",
        "set_no_tracking_fly_distance(1.50)",
        "set_no_tracking_a_param(0.900)",
        "set_no_tracking_c_param(0.100)",
        "set_no_tracking_time_coef(0.100)",
        "set_overflight_distance(0.080)",
        "set_overflight_samples(2)",
        "set_rotation_cor(0)",
        "set_min_rotation_cor(45)",
        "set_stop_spot_distance(8.0)",
        "set_stop_coef(0.80)",
        "set_video_recording(1)",
        "set_quality_hyst(30)",
        "set_time_limit_coef(1.30)",
        "set_reverse_brake_time(0.3)",
        "set_reverse_brake_power(20.0)",
        "set_reverse_brake_dist_a(0.3)",
        "set_reverse_brake_dist_b(0.1)",
        "set_rot_low_angle_speed(1.3)",
        f"set_min_speed({drone_config.speed:.2f})",
        "set_z_sliding_window(32)",
        "set_reverse_brake_time_c(0.67)",
        "set_reverse_brake_time_d(0.33)",
    ])
    return "\n".join(lines)


def render_txt(waypoints: List[Waypoint]) -> str:
    lines = ["# index x y z hold_time gimbal_pitch speed"]
    for i, wp in enumerate(waypoints, start=1):
        lines.append(f"{i} {wp.x:.6f} {wp.y:.6f} {wp.z:.6f} {wp.hold_time:.3f} {wp.gimbal_pitch:.2f} {wp.speed:.3f}")
    return "\n".join(lines)

# -----------------------------
# layout
# -----------------------------
# ====== UI styles ======
PANEL_STYLE = {
    "border": "1px solid #eaeaea",
    "borderRadius": "12px",
    "padding": "12px",
    "background": "white",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
}

SECTION_TITLE_STYLE = {
    "margin": "0 0 8px 0",
    "fontSize": "16px",
    "fontWeight": "600",
    "letterSpacing": "0.2px",
}

# --- button styles ---
REPORT_BTN_STYLE_ENABLED = {
    "display": "flex",
    "justifyContent": "center",
    "alignItems": "center",
    "margin": "10px auto 0",
    "padding": "10px 16px",
    "fontSize": "17px",
    "borderRadius": "10px",
    "fontWeight": 600,
    "background": "linear-gradient(180deg, #22c55e, #16a34a)",
    "color": "white",
    "border": "1px solid #15803d",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.06)",
    "cursor": "pointer",
    "textAlign": "center",
}

REPORT_BTN_STYLE_DISABLED = {
    "display": "flex",
    "justifyContent": "center",
    "alignItems": "center",
    "margin": "10px auto 0",
    "padding": "10px 16px",
    "fontSize": "17px",
    "borderRadius": "10px",
    "fontWeight": 600,
    "background": "#e5e7eb",
    "color": "#6b7280",
    "border": "1px solid #cbd5e1",
    "boxShadow": "none",
    "cursor": "not-allowed",
    "pointerEvents": "none",
    "textAlign": "center",
}



BUILD_BTN_STYLE = {
    "display": "block",
    "margin": "8px 0 0",
    "padding": "10px 16px",
    "minWidth": "200px",
    "borderRadius": "10px",
    "fontWeight": 600,
    "background": "linear-gradient(180deg, #22c55e, #16a34a)",
    "color": "white",
    "border": "1px solid #15803d",
    "textAlign": "center",
    "cursor": "pointer",
}

UPLOAD_STYLE = {
    "border": "1px dashed #cbd5e1",
    "borderRadius": "10px",
    "padding": "10px",
    "textAlign": "center",
    "cursor": "pointer",
    "background": "#fafafa",
}

def render_main_layout():
    return html.Div(
        [
            #---- header ----
            html.Div(
                className="header",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "auto 1fr auto",
                    "alignItems": "center",
                    "gap": "16px",
                    "padding": "12px 16px",
                    "borderBottom": "1px solid #eaeaea",
                    "position": "sticky",
                    "top": 0,
                    "background": "white",
                    "zIndex": 10,
                },
                children=[
                    #app logo (left side)
                    html.Img(
                        src=app_logo_src,
                        style={"height": "60px", "width": "auto"} if app_logo_src else {"display": "none"},
                        alt="App Logo",
                    ),

                    #title + description
                    html.Div(
                        [
                            html.H1(
                                "Fís - Target Flight Generator",
                                style={"margin": 0, "fontSize": "35px", "letterSpacing": "0.3px"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "One of the first platforms to provide a streamlined solution for indoor target flights paths generation, 3D scanning and reporting.",
                                        style={"marginBottom": "6px"},
                                    ),
                                    html.Div("Welcome to Fìs. Enjoy planning, flying, and reconstructing!"),
                                ],
                                style={"color": "#666", "fontSize": "20px", "maxWidth": "1100px"},
                            ),
                        ]
                    ),

                    #university logo (right side, clickable)
                    html.A(
                        html.Img(
                            src=logo_src,
                            style={"height": "40px", "width": "auto"} if logo_src else {"display": "none"},
                            alt="University Logo",
                        ),
                        href="https://www.ucd.ie/",
                        target="_blank",
                        style={"display": "inline-block"} if logo_src else {"display": "none"},
                    ),
                ],
            ),

            #===== main 2-col grid + modality row + downloads row =====
            html.Div(
                [
                    #LEFT CONTROLS
                    html.Div(
                        [
                            html.H3("Room setup & targeting", style=SECTION_TITLE_STYLE),

                            html.Label("Room dimensions (m)"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("W "),
                                            dcc.Input(
                                                id="room-w",
                                                type="number",
                                                value=ROOM_DIMS[0],
                                                min=0.5,
                                                step=0.1,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        style={"flex": "1", "minWidth": "80px", "marginRight": "6px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Span("L "),
                                            dcc.Input(
                                                id="room-l",
                                                type="number",
                                                value=ROOM_DIMS[1],
                                                min=0.5,
                                                step=0.1,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        style={"flex": "1", "minWidth": "80px", "marginRight": "6px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Span("H "),
                                            dcc.Input(
                                                id="room-h",
                                                type="number",
                                                value=ROOM_DIMS[2],
                                                min=0.5,
                                                step=0.1,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        style={"flex": "1", "minWidth": "80px"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "6px"},
                            ),

                            html.Br(),
                            html.Label("Clearances (m)"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Floor "),
                                            dcc.Input(
                                                id="floor-clear",
                                                type="number",
                                                value=0.0,
                                                min=0.0,
                                                step=0.05,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        style={"flex": "1", "minWidth": "120px", "marginRight": "6px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Ceiling "),
                                            dcc.Input(
                                                id="ceil-clear",
                                                type="number",
                                                value=0.0,
                                                min=0.0,
                                                step=0.05,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        style={"flex": "1", "minWidth": "120px"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "6px"},
                            ),

                            html.Br(),
                            html.Label("Granularity"),
                            dcc.RadioItems(
                                id="granularity",
                                options=[
                                    {"label": " Larger", "value": "larger"},
                                    {"label": " Smaller", "value": "smaller"},
                                ],
                                value="larger",
                                labelStyle={"display": "inline-block", "marginRight": "12px"},
                            ),

                            html.Br(),
                            dcc.Checklist(
                                id="show-labels",
                                options=[{"label": " Show cell IDs in 3D", "value": "show"}],
                                value=[],
                                style={"marginBottom": "10px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Select by ID"),
                                    dcc.Input(
                                        id="cube-id-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        debounce=True,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"marginTop": "10px"},
                            ),
                            html.Div(id="selected-info", style={"marginTop": "8px", "fontFamily": "monospace"}),

                            html.Br(),
                            html.Div(id="voxel-info", style={"color": "#555", "fontSize": "12px"}),
                        ],
                        style={**PANEL_STYLE, "gridArea": "leftControls"},
                    ),

                    #RIGHT CONTROLS
                    html.Div(
                        [
                            html.H3("Flight path parameters", style=SECTION_TITLE_STYLE),

                            #inputs grid
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Orbit radius (m)"),
                                            dcc.Input(
                                                id="orbit-radius",
                                                type="number",
                                                value=0.6,
                                                min=0.2,
                                                step=0.1,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Views (per orbit)"),
                                            dcc.Input(
                                                id="orbit-views",
                                                type="number",
                                                value=12,
                                                min=3,
                                                step=1,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Hover time per view (s)"),
                                            dcc.Input(
                                                id="orbit-dwell",
                                                type="number",
                                                value=2.0,
                                                min=0.0,
                                                step=0.5,
                                                debounce=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                                    "gap": "12px",
                                    "marginBottom": "10px",
                                },
                            ),

                            #flight summary card
                            html.Div(
                                id="flight-summary",
                                style={
                                    "marginTop": "8px",
                                    "padding": "10px",
                                    "borderRadius": "10px",
                                    "border": "1px solid #d0e7ff",
                                    "background": "#f7fbff",
                                    "lineHeight": "1.6",
                                    "fontSize": "inherit",
                                    "fontFamily": "inherit",
                                },
                                children=[
                                    html.H3("Flight summary", style=SECTION_TITLE_STYLE),
                                    html.Div("Select a cell to see the mission summary."),
                                ],
                            ),

                            #link that opens the report page in a NEW TAB
                            html.A(
                                "Open report workspace",
                                id="btn-report-link",
                                href="/report",
                                target="_blank",
                                style=REPORT_BTN_STYLE_DISABLED,
                            ),
                        ],
                        style={**PANEL_STYLE, "gridArea": "rightControls"},
                    ),

                    #LEFT GRAPH
                    html.Div(
                        [
                            dcc.Graph(
                                id="overview-graph",
                                figure=make_overview_figure(grid, False, None),
                                clear_on_unhover=True,
                            )
                        ],
                        style={**PANEL_STYLE, "gridArea": "leftGraph"},
                    ),

                    #RIGHT GRAPH
                    html.Div([dcc.Graph(id="zoom-graph")], style={**PANEL_STYLE, "gridArea": "rightGraph"}),

                    # ---- modality selector ----
                    html.Div(
                        [
                            dcc.RadioItems(
                                id="modality",
                                options=[
                                    {"label": " Single region", "value": "single"},
                                    {"label": " Multi region",  "value": "multi"},
                                ],
                                value="single",
                                inline=True,
                            ),
                            dcc.Store(id="modality-store", data="single"),
                        ],
                        style={
                            **PANEL_STYLE,
                            "gridArea": "modality",
                            "display": "flex",
                            "gap": "16px",
                            "alignItems": "center",
                            "justifyContent": "center",
                        },
                    ),

                    #DOWNLOADS ROW (single set of three buttons)
                    html.Div(
                        [
                            html.Button(
                                "Download CSV",
                                id="btn-csv",
                                n_clicks=0,
                                style={"padding": "8px 12px", "borderRadius": "8px"},
                            ),
                            html.Button(
                                "Download TXT",
                                id="btn-txt",
                                n_clicks=0,
                                style={"padding": "8px 12px", "borderRadius": "8px", "marginLeft": "8px"},
                            ),
                            html.Button(
                                "Download DPT",
                                id="btn-dpt",
                                n_clicks=0,
                                style={"padding": "8px 12px", "borderRadius": "8px", "marginLeft": "8px"},
                            ),
                            dcc.Download(id="dl-csv"),
                            dcc.Download(id="dl-txt"),
                            dcc.Download(id="dl-dpt"),
                        ],
                        style={
                            **PANEL_STYLE,
                            "gridArea": "downloads",
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "gap": "8px",
                            "flexWrap": "wrap",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(420px, 1fr) minmax(420px, 1fr)",
                    # ----add a "modality" row before downloads ----
                    "gridTemplateAreas": '"leftControls rightControls" '
                                         '"leftGraph rightGraph" '
                                         '"modality modality" '
                                         '"downloads downloads"',
                    "columnGap": "12px",
                    "rowGap": "12px",
                    "alignItems": "start",
                },
            ),

            #hidden stores (host)
            dcc.Store(id="grid-store"),
            dcc.Store(id="selected-id-store"),

            #---- add-on hidden stores (summary + export payloads) ----
            *multi_hidden_components(),
        ],
        style={"padding": "12px"},
    )


    

def render_report_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Flight Report Builder"),
                    html.Div("Upload the flight log and the intended flightpath, then build the report."),
                ],
                style={"marginBottom": "12px"}
            ),

            #uploads row
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Upload flight log (CSV)"),
                            dcc.Upload(
                                id="report-upload-log",
                                children=html.Div("Drag & drop or click to upload log.csv"),
                                multiple=False, accept=".csv,text/csv", style=UPLOAD_STYLE,
                            ),
                            html.Div(id="report-log-status", style={"fontSize":"12px", "color":"#6b7280", "marginTop":"6px"}),
                        ],
                        style={"display":"flex","flexDirection":"column","gap":"6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Upload flightpath (CSV)"),
                            dcc.Upload(
                                id="report-upload-path",
                                children=html.Div("Drag & drop or click to upload path.csv"),
                                multiple=False, accept=".csv,text/csv", style=UPLOAD_STYLE,
                            ),
                            html.Div(id="report-path-status", style={"fontSize":"12px", "color":"#6b7280", "marginTop":"6px"}),
                        ],
                        style={"display":"flex","flexDirection":"column","gap":"6px"},
                    ),
                ],
                style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}
            ),

            #hidden stores to keep the uploaded bytes
            dcc.Store(id="report-log-b64"),
            dcc.Store(id="report-path-b64"),

            #build button
            html.Button("Build report", id="btn-build-report", n_clicks=0, style=BUILD_BTN_STYLE),

            #error/status
            html.Div(id="report-error", style={"color":"#dc2626","marginTop":"8px"}),

            #rendered report (inline)
            html.Iframe(
                id="report-frame",
                srcDoc="",
                style={"width":"100%","height":"80vh","border":"1px solid #e5e7eb","borderRadius":"10px","marginTop":"12px"},
            ),
        ],
        style={"maxWidth":"1100px","margin":"0 auto","padding":"12px"}
    )



app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Target Flight Generator"
logo_path = pathlib.Path("/Users/stefaniaconte/Desktop/FlightGenerator_App/ucd_logo.jpg")
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
logo_src = f"data:image/jpeg;base64,{logo_b64}"
app_logo_path =pathlib.Path("/Users/stefaniaconte/Desktop/FlightGenerator_App/app_logo.png")
app_logo_b64 = base64.b64encode(app_logo_path.read_bytes()).decode()
app_logo_src = f"data:image/jpeg;base64,{app_logo_b64}"



app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-main",   children=render_main_layout()),
    html.Div(id="page-report", children=render_report_layout(), style={"display": "none"}),
])

mount_multi_region(
    app,
    grid_ref=grid,
    drone_ref=DRONE,
    generate_cube_orbit_fn=generate_cube_orbit,
    waypoint_cls=Waypoint,
    voxelgrid_cls=VoxelGrid,
)

# -----------------------------
#  helper functions
# -----------------------------

def _badge(condition: bool,
           ok_label: str = "OK",
           bad_label: str = "Not OK",
           ok_variant: str = "success",
           bad_variant: str = "danger"):
    """Return a coloured pill. If condition==True → ok_variant/ok_label else bad_variant/bad_label."""
    VARS = {
        "success": {"bg": "#f6ffed", "bd": "#b7eb8f", "fg": "#389e0d"},
        "warning": {"bg": "#fffbe6", "bd": "#ffe58f", "fg": "#d48806"},
        "danger":  {"bg": "#fff2f0", "bd": "#ffccc7", "fg": "#cf1322"},
        "info":    {"bg": "#e6f4ff", "bd": "#91caff", "fg": "#0958d9"},
    }
    variant = ok_variant if condition else bad_variant
    v = VARS[variant]
    return html.Span(
        ok_label if condition else bad_label,
        style={
            "display": "inline-block",
            "padding": "2px 10px",
            "borderRadius": "999px",
            "border": f"1px solid {v['bd']}",
            "background": v["bg"],
            "color": v["fg"],
            "fontWeight": 600,
            "fontSize": "12px",
        },
    )


def build_current_waypoints(selected_data, radius, n_views, dwell, floor_clear, ceil_clear) -> List[Waypoint]:
    if not selected_data or selected_data.get("id") is None:
        raise PreventUpdate
    sel_id = int(selected_data["id"])
    if sel_id < 1 or sel_id > grid.nx * grid.ny * grid.nz:
        raise PreventUpdate
    i, j, k = grid.id_to_ijk(sel_id)
    cx, cy, cz = grid.index_to_center(i, j, k)

    radius = 0.6 if (radius is None or radius <= 0) else float(radius)
    n_views = max(3, int(n_views or 12))
    dwell = float(dwell or 2.0)
    floor_clear = float(floor_clear or 0.0)
    ceil_clear = float(ceil_clear or 0.0)

    wps = generate_cube_orbit(
        drone=DRONE, dims=grid.dims, center=(cx, cy, cz),
        radius=radius, n_views=n_views, dwell_s=dwell,
        wall_clearance=0.02, floor_clearance=floor_clear, ceiling_clearance=ceil_clear
    )
    if not wps:
        raise PreventUpdate
    return wps



def validate_mission_simple(
    drone: DroneConfig,
    waypoints: List[Waypoint],
    ground_z: float = 0.0,
    fixed_overhead_s: float = 12.0,          
    wp_stop_s: float = 13.0,                 
    speed_override: float = 0.4,             
    vertical_speed_override: float = None,   
    battery_time_s: float = 1380.0,          
    battery_drainage: float = 4.0,           
) -> dict:
    """
    Minimal mission summary with per-waypoint stops, fixed overhead, takeoff/landing,
    and an extra battery drainage margin added in percentage points.
    Returns keys: num_waypoints, distance_m, total_time_s, battery_used_pct,
                  battery_ok, battery_warning, distance_ok, feasible
    """
    if not waypoints:
        return {
            "num_waypoints": 0,
            "distance_m": 0.0,
            "total_time_s": 0.0,
            "battery_used_pct": 0.0,
            "battery_ok": True,
            "battery_warning": False,
            "distance_ok": True,
            "feasible": True,
        }

    #speeds
    v_horiz = float(speed_override)
    v_vert  = float(vertical_speed_override) if vertical_speed_override is not None else v_horiz
    v_horiz = max(v_horiz, 1e-6)
    v_vert  = max(v_vert,  1e-6)

    #horizontal cruise distance
    horiz = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        horiz += math.hypot(b.x - a.x, b.y - a.y)

    #vertical takeoff/landing legs
    takeoff = abs(waypoints[0].z - ground_z)
    landing = abs(waypoints[-1].z - ground_z)

    total_dist = horiz + takeoff + landing

    #times
    travel_time = (horiz / v_horiz) + ((takeoff + landing) / v_vert)
    hover_time  = sum((getattr(wp, "hold_time", 0.0) or 0.0) for wp in waypoints)
    stops_time  = wp_stop_s * len(waypoints)   # +5s per waypoint
    total_time  = float(travel_time + hover_time + stops_time + fixed_overhead_s)

    #battery usage (time-based) + extra drainage margin
    T_batt = max(float(battery_time_s), 1e-6)
    battery_used_pct = 100.0 * (total_time / T_batt) + float(battery_drainage)

    #limits
    max_dist = float(getattr(drone, "max_distance", float("inf")))
    distance_ok = (total_dist <= max_dist)

    #battery checks
    battery_ok = (battery_used_pct <= 100.0)
    warn_thresh_pct = 100.0 * float(getattr(drone, "battery_warning_threshold", 0.85) or 0.85)
    battery_warning = (battery_used_pct >= warn_thresh_pct)

    feasible = battery_ok and distance_ok

    return {
        "num_waypoints": len(waypoints),
        "distance_m": total_dist,
        "total_time_s": total_time,
        "battery_used_pct": battery_used_pct,
        "battery_ok": battery_ok,
        "battery_warning": battery_warning,
        "distance_ok": distance_ok,
        "feasible": feasible,
    }



def fmt_duration(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(round(sec - 60*m))
    return f"{m}m {s}s" if m else f"{s}s"



# -----------------------------
# callbacks
# -----------------------------

#rebuild grid when room dims or granularity change
@app.callback(
    Output("grid-store", "data"),
    Output("voxel-info", "children"),
    Input("room-w", "value"),
    Input("room-l", "value"),
    Input("room-h", "value"),
    Input("granularity", "value"),
    prevent_initial_call=True
)
def rebuild_grid_from_dims(W, L, H, granularity):
    global grid
    W = float(W or ROOM_DIMS[0]); L = float(L or ROOM_DIMS[1]); H = float(H or ROOM_DIMS[2])
    dims = (max(0.5, W), max(0.5, L), max(0.5, H))

    target = TARGET_SMALLER_CELLS if granularity == "smaller" else TARGET_LARGER_CELLS
    nx, ny, nz = choose_equal_counts(dims, target_total_cubes=target)
    grid = VoxelGrid(dims, nx, ny, nz)

    info = f"Cell size: {grid.dx:.3f} × {grid.dy:.3f} × {grid.dz:.3f} m   |   Grid: {grid.nx} × {grid.ny} × {grid.nz} = {grid.nx*grid.ny*grid.nz}"
    return {"nx":grid.nx, "ny":grid.ny, "nz":grid.nz, "cell":[grid.dx, grid.dy, grid.dz], "dims":dims}, info

@app.callback(
    Output("overview-graph", "figure"),
    Input("show-labels", "value"),
    Input("selected-id-store", "data"),
    Input("grid-store", "data"),
    prevent_initial_call=False
)
def update_overview(show_labels_value, selected_data, _griddata):
    show_labels = ("show" in (show_labels_value or []))
    selected_id = selected_data["id"] if selected_data else None
    fig = make_overview_figure(grid, show_labels, selected_id)
    return fig

@app.callback(
    Output("selected-id-store", "data"),
    Output("cube-id-input", "value"),
    Output("selected-info", "children"),
    Input("overview-graph", "clickData"),
    Input("cube-id-input", "value"),
    State("selected-id-store", "data"),
    prevent_initial_call=True
)
def select_cube(clickData, typed_id, current_sel):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
    sel_id = None

    if trigger == "overview-graph" and clickData:
        point = clickData["points"][0]
        cd = point.get("customdata", None)
        if cd is not None and len(cd) >= 1:
            sel_id = int(cd[0])
    elif trigger == "cube-id-input" and typed_id:
        sel_id = int(typed_id)
    else:
        sel_id = (current_sel or {}).get("id", None)

    max_id = grid.nx * grid.ny * grid.nz
    if sel_id is not None:
        sel_id = int(max(1, min(max_id, sel_id)))

    info = ""
    if sel_id is not None:
        i,j,k = grid.id_to_ijk(sel_id)
        cx,cy,cz = grid.index_to_center(i,j,k)
        info = f"Selected: ID={sel_id}  (i,j,k)=({i},{j},{k})  center=({cx:.2f},{cy:.2f},{cz:.2f})"

    return {"id": sel_id} if sel_id is not None else None, sel_id, info

@app.callback(
    Output("zoom-graph", "figure"),
    Input("selected-id-store", "data"),
    Input("orbit-radius", "value"),
    Input("orbit-views", "value"),
    Input("orbit-dwell", "value"),
    Input("floor-clear", "value"),
    Input("ceil-clear", "value"),
    prevent_initial_call=False
)
def update_zoom(selected_data, radius, n_views, dwell, floor_clear, ceil_clear):
    sel_id = selected_data["id"] if selected_data else None
    radius = 0.6 if (radius is None or radius <= 0) else float(radius)
    n_views = max(3, int(n_views or 12))
    dwell = float(dwell or 2.0)
    floor_clear = float(floor_clear or 0.0)
    ceil_clear = float(ceil_clear or 0.0)
    return make_zoom_figure(grid, DRONE, sel_id, radius, n_views, dwell, floor_clear, ceil_clear)


@app.callback(
    Output("dl-csv", "data"),
    Output("dl-txt", "data"),
    Output("dl-dpt", "data"),
    Input("btn-csv", "n_clicks"),
    Input("btn-txt", "n_clicks"),
    Input("btn-dpt", "n_clicks"),
    State("modality-store", "data"),
    State("selected-id-store", "data"),
    State("orbit-radius", "value"),
    State("orbit-views", "value"),
    State("orbit-dwell", "value"),
    State("floor-clear", "value"),
    State("ceil-clear", "value"),
    State("multi-export-payloads", "data"),   # <-- from add-on (ready-to-use dicts)
    prevent_initial_call=True
)
def unified_download(n_csv, n_txt, n_dpt,
                     modality, selected_data, radius, n_views, dwell, floor_clear, ceil_clear,
                     multi_payloads):
    t = (callback_context.triggered[0]["prop_id"].split(".")[0]
         if callback_context.triggered else None)

    csv_out = txt_out = dpt_out = no_update

    if modality == "multi":
        payloads = multi_payloads or {}
        if t == "btn-csv":
            csv_out = payloads.get("csv", no_update)
        elif t == "btn-txt":
            txt_out = payloads.get("txt", no_update)
        elif t == "btn-dpt":
            dpt_out = payloads.get("dpt", no_update)
        return csv_out, txt_out, dpt_out

    # ---- single region path ----
    #build_current_waypoints raises PreventUpdate if not ready
    wps = build_current_waypoints(selected_data, radius, n_views, dwell, floor_clear, ceil_clear)
    sel_id = (selected_data or {}).get("id", "path")

    if t == "btn-csv":
        return dcc.send_string(render_marvelmind_csv(wps), filename=f"flight_voxel_{sel_id}.csv"), no_update, no_update
    if t == "btn-txt":
        return no_update, dcc.send_string(render_txt(wps), filename=f"flight_voxel_{sel_id}.txt"), no_update
    if t == "btn-dpt":
        return no_update, no_update, dcc.send_string(render_dpt(wps, DRONE), filename=f"flight_voxel_{sel_id}.dpt")

    raise PreventUpdate



@app.callback(
    Output("btn-csv", "disabled"),
    Output("btn-txt", "disabled"),
    Output("btn-dpt", "disabled"),
    Input("modality-store", "data"),
    Input("selected-id-store", "data"),
    Input("orbit-radius", "value"),
    Input("orbit-views", "value"),
    Input("orbit-dwell", "value"),
    Input("floor-clear", "value"),
    Input("ceil-clear", "value"),
    Input("multi-export-payloads", "data"),
)
def toggle_downloads(modality, selected_data, radius, n_views, dwell, floor_clear, ceil_clear, multi_payloads):
    if modality == "multi":
        ready = bool(multi_payloads and any(multi_payloads.get(k) for k in ("csv", "txt", "dpt")))
        disabled = not ready
        return disabled, disabled, disabled

    #single
    try:
        _ = build_current_waypoints(selected_data, radius, n_views, dwell, floor_clear, ceil_clear)
        return False, False, False
    except PreventUpdate:
        return True, True, True

    
@app.callback(
    Output("page-main", "style"),
    Output("page-report", "style"),
    Input("url", "pathname")
)
def _toggle_pages(pathname):
    show_report = str(pathname).rstrip("/") == "/report"
    return (
        {"display": "none"} if show_report else {"display": "block"},
        {"display": "block"} if show_report else {"display": "none"},
    )


@app.callback(
    Output("flight-summary", "children"),
    Output("flight-summary", "style"),
    
    Input("modality-store", "data"),
    Input("multi-flight-summary-json", "data"),
    
    Input("selected-id-store", "data"),
    Input("orbit-radius", "value"),
    Input("orbit-views", "value"),
    Input("orbit-dwell", "value"),
    Input("floor-clear", "value"),
    Input("ceil-clear", "value"),
)
def update_flight_summary_router(modality, multi_summary_json,
                                 selected_data, radius, n_views, dwell, floor_clear, ceil_clear):
    base_style = {
        "marginTop": "8px",
        "padding": "10px",
        "borderRadius": "10px",
        "border": "1px solid #d0e7ff",
        "background": "#f7fbff",
        "lineHeight": "1.6",
        "fontSize": "inherit",
        "fontFamily": "inherit",
    }

    def render_common_summary(title: str, stats: dict):
        """Render the SAME fields for single & multi using your helpers."""
        feasible   = bool(stats.get("feasible", True))
        batt_ok    = bool(stats.get("battery_ok", True))
        batt_warn  = bool(stats.get("battery_warning", False))
        dist_ok    = bool(stats.get("distance_ok", True))
        n_wps      = int(stats.get("total_waypoints", stats.get("num_waypoints", 0)))
        t_total_s  = float(stats.get("total_time_s", 0.0))
        batt_pct   = float(stats.get("battery_used_pct", 0.0))
        dist_m     = float(stats.get("distance_m", 0.0))

        #same coloring rules as single
        if not feasible:
            style = {**base_style, "background": "#fff2f0", "border": "1px solid #ffccc7"}  # red
            status = "Not feasible"
        elif batt_warn:
            style = {**base_style, "background": "#fffbe6", "border": "1px solid #ffe58f"}  # yellow
            status = "Feasible (battery warning)"
        else:
            style = base_style
            status = "Feasible"

        status_line = html.Div(
            [
                html.Span("•  Battery: ", style={"fontWeight": 600}),
                _badge(batt_ok, "OK", "Insufficient", ok_variant="success", bad_variant="danger"),

                html.Span("•  Warning: ", style={"fontWeight": 600}),
                _badge(not batt_warn, "NO", "YES", ok_variant="success", bad_variant="warning"),

                html.Span("•  Distance: ", style={"fontWeight": 600}),
                _badge(dist_ok, "OK", "Limit exceeded", ok_variant="success", bad_variant="danger"),
            ],
            style={"gridColumn": "1 / -1", "display": "flex", "gap": "8px",
                   "alignItems": "center", "flexWrap": "wrap"},
        )

        label = {"fontWeight": 600}
        extra = []
        #show Regions when provided (multi mode)
        if "regions" in stats and stats.get("regions") is not None:
            extra.append(
                html.Div([html.Span("Regions: ", style=label),
                          html.Span(str(int(stats.get("regions", 0))))])
            )

        children = [
            html.H3(title, style=SECTION_TITLE_STYLE),
            html.Div([html.Span("Status: ", style=label), html.Span(status)]),
            status_line,
            *extra,
            html.Div([html.Span("Waypoints: ", style=label), html.Span(f"{n_wps}")]),
            html.Div([html.Span("Total flight time: ", style=label), html.Span(fmt_duration(t_total_s))]),
            html.Div([html.Span("Battery used: ", style=label), html.Span(f"{batt_pct:.1f}%")]),
            html.Div([html.Span("Distance: ", style=label), html.Span(f"{dist_m:.1f} m")]),
        ]
        return children, style

    #---------- MULTI ----------
    if modality == "multi":
        if not multi_summary_json:
            return (
                [
                    html.H3("Multi-region flight summary", style=SECTION_TITLE_STYLE),
                    html.Div("Click “Preview region” in the multi panel to compute the multi-region plan."),
                ],
                {**base_style, "background": "#f5f5f5", "border": "1px solid #eaeaea"},
            )
        stats = multi_summary_json.get("stats") or {}
        return render_common_summary("Multi-region flight summary", stats)

    #---------- SINGLE ----------
    try:
        wps = build_current_waypoints(selected_data, radius, n_views, dwell, floor_clear, ceil_clear)
    except Exception:
        return (
            [
                html.H3("Flight summary", style=SECTION_TITLE_STYLE),
                html.Div("Select a cell to see the mission summary."),
            ],
            {**base_style, "background": "#f5f5f5", "border": "1px solid #eaeaea"},
        )

    metrics = validate_mission_simple(DRONE, wps, ground_z=0.0, fixed_overhead_s=12.0)
    
    stats_single = {
        "regions": None,  # N/A in single mode
        "total_waypoints": metrics["num_waypoints"],
        "distance_m": metrics["distance_m"],
        "total_time_s": metrics["total_time_s"],
        "battery_used_pct": metrics["battery_used_pct"],
        "battery_ok": metrics["battery_ok"],
        "battery_warning": metrics["battery_warning"],
        "distance_ok": metrics["distance_ok"],
        "feasible": metrics["feasible"],
    }
    return render_common_summary("Flight summary", stats_single)



@app.callback(
    Output("btn-report-link", "style"),
    Input("btn-csv", "n_clicks"),
    Input("btn-txt", "n_clicks"),
    Input("btn-dpt", "n_clicks"),
)
def enable_report_link(n_csv, n_txt, n_dpt):
    clicked_any = (n_csv or 0) > 0 or (n_txt or 0) > 0 or (n_dpt or 0) > 0
    return REPORT_BTN_STYLE_ENABLED if clicked_any else REPORT_BTN_STYLE_DISABLED

@app.callback(
    Output("report-log-b64",  "data"),
    Output("report-log-status","children"),
    Input("report-upload-log", "contents"),
    State("report-upload-log", "filename"),
    prevent_initial_call=True
)
def handle_report_log(contents, filename):
    if not contents:
        return no_update, no_update
    b64 = _b64_from_upload(contents)
    name = filename or "log.csv"
    return b64, html.Span([html.Span("Uploaded: ", style={"fontWeight":600}), html.Span(name)], style={"color":"#16a34a"})

@app.callback(
    Output("report-path-b64",  "data"),
    Output("report-path-status","children"),
    Input("report-upload-path", "contents"),
    State("report-upload-path", "filename"),
    prevent_initial_call=True
)
def handle_report_path(contents, filename):
    if not contents:
        return no_update, no_update
    b64 = _b64_from_upload(contents)
    name = filename or "path.csv"
    return b64, html.Span([html.Span("Uploaded: ", style={"fontWeight":600}), html.Span(name)], style={"color":"#16a34a"})


@app.callback(
    Output("report-frame", "srcDoc"),
    Output("report-error", "children"),
    Input("btn-build-report", "n_clicks"),
    State("report-log-b64", "data"),
    State("report-path-b64", "data"),
    State("room-w", "value"),
    State("room-l", "value"),
    State("room-h", "value"),
    prevent_initial_call=True
)
def build_report(n, log_b64, path_b64, W, L, H):
    if not n:
        return no_update, no_update
    if not log_b64 or not path_b64:
        return no_update, "Please upload both files before building the report."

    try:
        log_bytes  = base64.b64decode(log_b64)
        path_bytes = base64.b64decode(path_b64)

        workspace = None
        if W and L and H:
            workspace = {"x": (0.0, float(W)), "y": (0.0, float(L)), "z": (0.0, float(H))}

        bundle = generate_report_bundle(log_bytes, path_bytes, workspace_dims=workspace)
        html_text = _report_html_from_bundle(bundle, title="Flight Report")
        return html_text, ""   
    except Exception as e:
        return no_update, f"Error generating report: {e}"

@app.callback(
    Output("modality-store", "data"),
    Input("modality", "value"),
    prevent_initial_call=False
)
def _sync_modality_store(v):
    return v or "single"


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Starting app at http://127.0.0.1:8050 ...")
    app.run_server(debug=True)
