import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
import numpy as np


@dataclass
class DroneConfig:
    """
    holds drone settings and limits
    """
    weight: float                    # weight in kg
    max_battery_time: float          # max flight time in seconds - tested (unused here)
    max_distance: float              # max travel distance in meters
    horizontal_fov: float            # camera horizontal field of view in degrees
    vertical_fov: float              # camera vertical field of view in degrees
    fps: float                       # camera frame rate in frames per second
    resolution: Tuple[int, int]      # camera resolution in pixels (width, height) (unused here)
    speed: float                     # nominal flight speed in m/s (unused here for scan geometry)
    min_altitude: float              # minimum safe flight altitude in meters
    hover_buffer: float              # extra hover time for stabilization in seconds (used for hold)
    battery_warning_threshold: float # threshold for battery warning as a fraction of max time (default 80%)

@dataclass
class Waypoint:
    """
    stores a single waypoint for the mission
    """
    x: float
    y: float
    z: float
    gimbal_pitch: float
    speed: float
    hold_time: float


def calculate_footprint(drone: DroneConfig, distance: float) -> Tuple[float, float]:
    hfov = math.radians(drone.horizontal_fov)
    vfov = math.radians(drone.vertical_fov)
    w = 2 * distance * math.tan(hfov / 2)
    h = 2 * distance * math.tan(vfov / 2)
    return w, h


def calculate_speed(footprint_len: float, overlap: float, fps: float) -> float:
    return footprint_len * (1 - overlap) * fps


def generate_spiral_scan(
        drone: DroneConfig,
        dims: Tuple[float, float, float],
        altitude: float,
        overlap: float,
        wall_offset: float,
        clearance: float,
        target_turns: int = 3,
) -> List[Waypoint]:
    """
    generating a straight-edged outward rectangular spiral of waypoints over a planar area.

    The spiral starts at the center of the inset rectangle (accounting for wall_offset + clearance on X)
    and just clearance on Y, and expands outward in the order: right, up, left, down, with layer lengths
    increasing by one every two directions. The step size respects camera coverage and produces roughly
    `target_turns` layers before hitting the bounds.
    """
    # separate X-margin (wall + clearance) from Y-padding (just clearance)
    inset_x = wall_offset + clearance
    inset_y = clearance
    hold = 1.0 / drone.fps + drone.hover_buffer

    min_x = inset_x
    max_x = dims[0] - inset_x
    min_y = inset_y
    max_y = dims[1] - inset_y

    if min_x >= max_x or min_y >= max_y:
        return []

    altitude = max(clearance, drone.min_altitude)

    # footprint-based nominal step
    fp_long, fp_short = calculate_footprint(drone, altitude)
    avg_fp = (fp_long + fp_short) / 2.0
    nominal_step = avg_fp * (1 - overlap)
    speed = calculate_speed(avg_fp, overlap, drone.fps)

    # determine step for roughly target_turns layers
    span_x = max_x - min_x
    span_y = max_y - min_y
    min_span = min(span_x, span_y)
    step = min(nominal_step, min_span / (2 * target_turns + 1))

    # start at center
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    x, y = cx, cy

    wps: List[Waypoint] = [Waypoint(x, y, altitude, 0.0, speed, hold)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    layer = 1
    dir_idx = 0

    # outward spiral
    while True:
        moved = False
        for _ in range(2):
            dx, dy = directions[dir_idx % 4]
            for _ in range(layer):
                x_new = x + dx * step
                y_new = y + dy * step
                if not (min_x <= x_new <= max_x and min_y <= y_new <= max_y):
                    return wps
                x, y = x_new, y_new
                wps.append(Waypoint(x, y, altitude, 0.0, speed, hold))
                moved = True
            dir_idx += 1
        layer += 1
        if not moved:
            break
    return wps


def reduce_waypoints_collinear(waypoints: List[Waypoint], eps=1e-6) -> List[Waypoint]:
    if len(waypoints) <= 2:
        return waypoints.copy()
    reduced = [waypoints[0]]
    for prev, curr, nxt in zip(waypoints, waypoints[1:], waypoints[2:]):
        v1 = (curr.x - prev.x, curr.y - prev.y, curr.z - prev.z)
        v2 = (nxt.x - curr.x, nxt.y - curr.y, nxt.z - curr.z)
        cross = (v1[1]*v2[2] - v1[2]*v2[1],
                 v1[2]*v2[0] - v1[0]*v2[2],
                 v1[0]*v2[1] - v1[1]*v2[0])
        cross_norm = math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        if cross_norm < eps and dot > 0:
            continue
        reduced.append(curr)
    reduced.append(waypoints[-1])
    return reduced

def stacked_spiral(heights: Tuple[float], base_waypoints: List[Waypoint]) -> List[Waypoint]:
    ''''stacks 2d spirals but reverting order: 
    - first tier goes from inside to outside
    - second goes from outside to outside 
    - and so forth 
    In this way the drone is able to start for 
    the same point of the sequential tier, just with a height increase
    - for each loop the height should be +0.2m.'''
    if not heights or not base_waypoints:
        return []

    def clone_with_z(seq: List[Waypoint], z: float) -> List[Waypoint]:
        return [Waypoint(w.x, w.y, z, w.gimbal_pitch, w.speed, w.hold_time) for w in seq]

    out: List[Waypoint] = []

    # Precompute forward/reverse views of the base 2D spiral
    forward = base_waypoints
    reverse = list(reversed(base_waypoints))

    # Layer 0: forward at heights[0]
    current_layer = clone_with_z(forward, heights[0])
    out.extend(current_layer)

    # Convenience XYs for exact comparisons (identical floats due to cloning)
    base_start_xy = (forward[0].x, forward[0].y)
    base_end_xy   = (forward[-1].x, forward[-1].y)

    # Subsequent layers
    for z in heights[1:]:
        prev_last_xy = (out[-1].x, out[-1].y)

        if prev_last_xy == base_end_xy:
            oriented = reverse      # next must start at base[-1]
        elif prev_last_xy == base_start_xy:
            oriented = forward      # next must start at base[0]
        else:
            # Fallback (shouldn't happen with the above pattern):
            # choose whichever starts at prev_last_xy if present, else default to reverse.
            if (oriented := reverse)[0].x == prev_last_xy[0] and oriented[0].y == prev_last_xy[1]:
                pass
            elif forward[0].x == prev_last_xy[0] and forward[0].y == prev_last_xy[1]:
                oriented = forward
            else:
                oriented = reverse

        # Append this layer at new height; first point shares XY with previous last (vertical move)
        out.extend(clone_with_z(oriented, z))

    return out


def visualize_waypoints_2d(
    waypoints: List['Waypoint'],
    dims: Tuple[float, float, float],  # (room_width, room_length, room_height)
    draw_lines: bool = False,
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None,
    axis_limits: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    equal_aspect_top: bool = True,
    invert: Optional[Dict[str, Dict[str, bool]]] = None
):
    """
    Plot top-down, side, and front views of waypoints, with optional fixed per-view axis limits.

    axis_limits:
      {
        'top':  {'x': (0, 6), 'y': (0, 8)},
        'side': {'x': (0, 6), 'y': (0, 3)},
        'front':{'x': (0, 8), 'y': (0, 3)}
      }

    invert example (to flip axes if you want y downwards, etc.):
      {
        'top':  {'x': False, 'y': True},
        'side': {'x': False, 'y': False},
        'front':{'x': False, 'y': False}
      }
    """
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    w, l, h = dims

    line_kwargs   = line_kwargs   or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.7, 'color': 'C2'}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.8, 'color': 'C1'}
    invert = invert or {}

    # Defaults from room dimensions (what you already had)
    default_limits = {
        'top':  {'x': (0, w), 'y': (0, l)},
        'side': {'x': (0, w), 'y': (0, h)},
        'front':{'x': (0, l), 'y': (0, h)}
    }
    # Merge user limits over defaults
    if axis_limits:
        for view in ('top','side','front'):
            if view in axis_limits:
                for axkey in ('x','y'):
                    if axkey in axis_limits[view]:
                        default_limits[view][axkey] = axis_limits[view][axkey]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    views = [
        ('top',   xs, ys, 'top-down', ('x (m)', 'y (m)')),
        ('side',  xs, zs, 'side',     ('x (m)', 'z (m)')),
        ('front', ys, zs, 'front',    ('y (m)', 'z (m)'))
    ]

    for ax, (key, X, Y, title, (xlabel, ylabel)) in zip(axs, views):
        if draw_lines:
            ax.plot(X, Y, **line_kwargs)
        ax.scatter(X, Y, **marker_kwargs)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True)

        # Apply limits (fixed)
        xlim = default_limits[key]['x']
        ylim = default_limits[key]['y']
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Aspect
        if key == 'top' and equal_aspect_top:
            ax.set_aspect('equal', 'box')

        # Optional axis inversion
        if invert.get(key, {}).get('x', False):
            ax.invert_xaxis()
        if invert.get(key, {}).get('y', False):
            ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

def visualize_waypoints_3d(
    waypoints: List['Waypoint'],
    elev: float = 30,
    azim: float = 45,
    draw_lines: bool = True,
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None,
    # New:
    dims: Optional[Tuple[float, float, float]] = None,  # (width, length, height)
    axis_limits: Optional[Dict[str, Tuple[float, float]]] = None,  # {'x':(xmin,xmax),'y':(...), 'z':(...)}
    invert: Optional[Dict[str, bool]] = None,  # {'x':True/False, 'y':..., 'z':...}
    equal_aspect: bool = True
):
    """
    3D view of the path with optional fixed axes.

    Examples:
      # Use room dims:
      visualize_waypoints_3d(waypoints, dims=(6, 9, 3))

      # Or explicit limits:
      visualize_waypoints_3d(
          waypoints,
          axis_limits={'x': (0, 6), 'y': (0, 9), 'z': (0, 3)}
      )
    """
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]

    line_kwargs   = line_kwargs   or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.8, 'color': 'C2'}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.9, 'color': 'C1'}
    invert = invert or {}

    # Establish defaults from dims (if given), otherwise from data extent
    if dims is not None:
        w, l, h = dims
        default_limits = {'x': (0.0, w), 'y': (0.0, l), 'z': (0.0, h)}
    else:
        # Data-driven fallback with a small margin
        def _lim(v):
            vmin, vmax = min(v), max(v)
            if vmin == vmax:  # degenerate (all same value)
                vmin -= 0.5
                vmax += 0.5
            pad = 0.02 * (vmax - vmin)
            return (vmin - pad, vmax + pad)
        default_limits = {'x': _lim(xs), 'y': _lim(ys), 'z': _lim(zs)}

    # Overwrite with explicit axis_limits if provided
    if axis_limits:
        for k in ('x', 'y', 'z'):
            if k in axis_limits and axis_limits[k] is not None:
                default_limits[k] = axis_limits[k]

    # Build plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if draw_lines:
        ax.plot(xs, ys, zs, **line_kwargs)
    ax.scatter(xs, ys, zs, **marker_kwargs)

    ax.set(title='3D view', xlabel='x (m)', ylabel='y (m)')
    ax.set_zlabel('z (m)')

    # Apply fixed limits
    ax.set_xlim(*default_limits['x'])
    ax.set_ylim(*default_limits['y'])
    ax.set_zlim(*default_limits['z'])

    # Equal aspect so room proportions look right
    if equal_aspect:
        xr = default_limits['x'][1] - default_limits['x'][0]
        yr = default_limits['y'][1] - default_limits['y'][0]
        zr = default_limits['z'][1] - default_limits['z'][0]
        try:
            # Matplotlib 3.3+
            ax.set_box_aspect((xr, yr, zr))
        except Exception:
            # Fallback: cube around the data center
            maxrange = max(xr, yr, zr)
            xc = sum(default_limits['x']) / 2
            yc = sum(default_limits['y']) / 2
            zc = sum(default_limits['z']) / 2
            half = maxrange / 2
            ax.set_xlim(xc - half, xc + half)
            ax.set_ylim(yc - half, yc + half)
            ax.set_zlim(zc - half, zc + half)

    # Optional axis inversion
    if invert.get('x'): ax.invert_xaxis()
    if invert.get('y'): ax.invert_yaxis()
    if invert.get('z'): ax.invert_zaxis()

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()

def validate_mission(
    drone: DroneConfig,
    waypoints: List[Waypoint],
    include_takeoff_landing: bool = True,
    ground_z: float = 0.0,
    controller_wp_pause_s: float = 2.0,   # from set_waypoint_pause(2.0) in your export
    fixed_pre_post_pauses_s: float = 12.0 # pause(4)+pause(1)+pause(1)+pause(6) in your export
) -> Tuple[bool, dict]:
    if not waypoints:
        return False, {
            'distance': 0.0, 'distance_takeoff': 0.0, 'distance_cruise': 0.0, 'distance_landing': 0.0,
            'num_corners': 0, 't90_used_s': 0.0, 'turn_time_90s': 0.0,
            'travel_time': 0.0, 'hover_time': 0.0, 'controller_pause_time': 0.0,
            'fixed_overhead_time': 0.0, 'total_time': 0.0,
            'battery_usage_percent': 0.0, 'battery_ok': True, 'battery_warning': False, 'distance_ok': True
        }

    v_horiz = drone.speed
    v_vert  = getattr(drone, 'vertical_speed', None) or drone.speed

    # 90° yaw time from your settings: set_power_rot(46.67) vs nominal 60 -> ~93.3 deg/s
    base_power = 60.0
    base_yaw_deg_s = 120.0
    power_rot = 46.67
    yaw_rate_large = base_yaw_deg_s * (power_rot / base_power)
    t90 = 90.0 / max(yaw_rate_large, 1e-6)  # ~0.96 s

    total_distance = 0.0
    travel_time = 0.0

    # Takeoff (vertical)
    distance_takeoff = 0.0
    if include_takeoff_landing and waypoints[0].z != ground_z:
        distance_takeoff = abs(waypoints[0].z - ground_z)
        travel_time += distance_takeoff / v_vert
        total_distance += distance_takeoff

    # Cruise (2D, using drone.speed)
    distance_cruise = 0.0
    leg_vecs = []
    for wp1, wp2 in zip(waypoints, waypoints[1:]):
        dx, dy = wp2.x - wp1.x, wp2.y - wp1.y
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            continue
        d2 = math.hypot(dx, dy)
        distance_cruise += d2
        total_distance  += d2
        travel_time     += d2 / max(v_horiz, 1e-9)   # <-- this is the line you asked about
        leg_vecs.append((dx, dy))

    # Turn time: count ~90° corners via dot product
    def is_right_angle(v1, v2, tol_deg=15.0):
        n1 = math.hypot(*v1); n2 = math.hypot(*v2)
        if n1 < 1e-12 or n2 < 1e-12: return False
        cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.degrees(math.acos(cosang))
        return (90.0 - tol_deg) <= ang <= (90.0 + tol_deg)

    num_corners = sum(1 for v1, v2 in zip(leg_vecs, leg_vecs[1:]) if is_right_angle(v1, v2))
    turn_time = num_corners * t90
    travel_time += turn_time

    # Landing (vertical)
    distance_landing = 0.0
    if include_takeoff_landing and waypoints[-1].z != ground_z:
        distance_landing = abs(waypoints[-1].z - ground_z)
        travel_time += distance_landing / v_vert
        total_distance += distance_landing

    # Hover + controller pauses + fixed export pauses
    hover_time = sum(getattr(wp, 'hold_time', 0.0) for wp in waypoints)
    controller_pause_time = controller_wp_pause_s * len(waypoints)
    fixed_overhead_time   = fixed_pre_post_pauses_s

    total_time = travel_time + hover_time + controller_pause_time + fixed_overhead_time
    battery_used_pct = (total_time / drone.max_battery_time) * 100.0

    battery_ok = battery_used_pct <= 100.0
    battery_warning = battery_used_pct >= (drone.battery_warning_threshold * 100.0)
    distance_ok = total_distance <= drone.max_distance
    feasible = battery_ok and distance_ok

    return feasible, {
        'distance': total_distance,
        'distance_takeoff': distance_takeoff,
        'distance_cruise': distance_cruise,
        'distance_landing': distance_landing,
        'num_corners': num_corners,
        't90_used_s': t90,
        'turn_time_90s': turn_time,
        'travel_time': travel_time,
        'hover_time': hover_time,
        'controller_pause_time': controller_pause_time,
        'fixed_overhead_time': fixed_overhead_time,
        'total_time': total_time,
        'battery_usage_percent': battery_used_pct,
        'battery_ok': battery_ok,
        'battery_warning': battery_warning,
        'distance_ok': distance_ok
    }




def validate_mission_with_layers(
    drone: DroneConfig,
    waypoints: List[Waypoint],
    include_takeoff_landing: bool = True,
    ground_z: float = 0.0,
    controller_wp_pause_s: float = 2.0,   # from set_waypoint_pause(2.0)
    fixed_pre_post_pauses_s: float = 12.0 # pause(4)+pause(1)+pause(1)+pause(6)
) -> Tuple[bool, dict]:
    """
    3D stacked-spiral time estimator:
      - In-layer legs (constant Z): 2D distance / drone.speed
      - Vertical seams between layers: time from horizontal+vertical components
      - Takeoff/Landing: |Δz| / vertical_speed
      - Turns: count ~90° corners per layer, add t90 each (derived from set_power_rot=46.67)
      - Hover: sum(hold_time)
      - Controller pauses & fixed pre/post pauses included
    """
    if not waypoints:
        empty = {
            'distance': 0.0,
            'distance_takeoff': 0.0,
            'distance_cruise': 0.0,
            'distance_landing': 0.0,
            'travel_time': 0.0,
            'hover_time': 0.0,
            'controller_pause_time': 0.0,
            'fixed_overhead_time': 0.0,
            'total_time': 0.0,
            'battery_usage_percent': 0.0,
            'battery_ok': True,
            'battery_warning': False,
            'distance_ok': True,
            # New:
            'vertical_seams_count': 0,
            'vertical_seams_distance': 0.0,
            'vertical_seams_time': 0.0,
            'num_corners': 0,
            't90_used_s': 0.0,
            'turn_time_90s': 0.0,
            'layers': []
        }
        return False, empty

    v_horiz = drone.speed
    v_vert  = getattr(drone, 'vertical_speed', None) or drone.speed
    eps_z   = 1e-9

    # 90° yaw time from your settings: set_power_rot(46.67) vs nominal 60 -> ~93.3 deg/s
    base_power = 60.0
    base_yaw_deg_s = 120.0
    power_rot = 46.67
    yaw_rate_large = base_yaw_deg_s * (power_rot / base_power)  # ~93.3 deg/s
    t90 = 90.0 / max(yaw_rate_large, 1e-6)                      # ~0.96 s

    # ---- Build layers (contiguous runs of ~equal Z) ----
    layers: List[Tuple[float,int,int]] = []
    start = 0
    current_z = waypoints[0].z
    for i in range(1, len(waypoints)):
        if abs(waypoints[i].z - current_z) > eps_z:
            layers.append((current_z, start, i-1))
            start = i
            current_z = waypoints[i].z
    layers.append((current_z, start, len(waypoints)-1))

    # Per-layer stats scaffolding
    layer_stats = [{
        'z': z,
        'idx_start': s,
        'idx_end': e,
        'num_waypoints': (e - s + 1),
        'distance_2d': 0.0,
        'distance_3d': 0.0,
        'travel_time': 0.0,     # in-layer travel time (2D legs only; turns added later)
        'hover_time': sum(getattr(wp, 'hold_time', 0.0) for wp in waypoints[s:e+1]),
        'num_corners': 0,
        'turn_time_90s': 0.0,
        'total_time': 0.0,
        'battery_usage_percent': 0.0
    } for (z, s, e) in layers]

    def which_layer(idx: int) -> int:
        for li, (_, s, e) in enumerate(layers):
            if s <= idx <= e:
                return li
        return -1

    # ---- Helpers ----
    def is_right_angle(v1, v2, tol_deg=15.0) -> bool:
        n1 = math.hypot(v1[0], v1[1]); n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-12 or n2 < 1e-12: return False
        cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.degrees(math.acos(cosang))
        return (90.0 - tol_deg) <= ang <= (90.0 + tol_deg)

    # ---- Totals ----
    total_distance = 0.0     # 3D path length
    distance_takeoff = 0.0
    distance_cruise = 0.0    # sum of 2D in-layer distances + 3D seam distances
    distance_landing = 0.0

    travel_time = 0.0
    hover_time = sum(getattr(wp, 'hold_time', 0.0) for wp in waypoints)

    # ---- Takeoff ----
    if include_takeoff_landing and abs(waypoints[0].z - ground_z) > eps_z:
        distance_takeoff = abs(waypoints[0].z - ground_z)
        travel_time += distance_takeoff / v_vert
        total_distance += distance_takeoff

    # ---- Legs (classify in-layer vs seam) ----
    vertical_seams_count = 0
    vertical_seams_distance = 0.0  # 3D seam length
    vertical_seams_time = 0.0

    # Collect per-layer 2D vectors to count corners
    layer_leg_vecs: Dict[int, List[Tuple[float,float]]] = {i: [] for i in range(len(layers))}

    for i, (wp1, wp2) in enumerate(zip(waypoints, waypoints[1:])):
        dx, dy, dz = wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z
        d2 = math.hypot(dx, dy)
        d3 = math.sqrt(dx*dx + dy*dy + dz*dz)

        li1 = which_layer(i)
        li2 = which_layer(i+1)

        # Always accumulate total 3D distance
        total_distance += d3

        if li1 == li2 and li1 != -1 and abs(dz) <= eps_z:
            # In-layer horizontal leg (constant Z)
            distance_cruise += d2
            layer_stats[li1]['distance_2d'] += d2
            layer_stats[li1]['distance_3d'] += d3
            t_seg = d2 / max(v_horiz, 1e-9)
            layer_stats[li1]['travel_time'] += t_seg
            travel_time += t_seg
            if d2 > 1e-12:
                layer_leg_vecs[li1].append((dx, dy))
        else:
            # Seam between layers (usually vertical; allow diagonal just in case)
            vertical_seams_count += 1
            vertical_seams_distance += d3
            distance_cruise += d3
            # Time: decompose into horizontal + vertical components
            t_seam = 0.0
            if d2 > 0.0:
                t_seam += d2 / max(v_horiz, 1e-9)
            if abs(dz) > 0.0:
                t_seam += abs(dz) / max(v_vert, 1e-9)
            vertical_seams_time += t_seam
            travel_time += t_seam

    # ---- Landing ----
    if include_takeoff_landing and abs(waypoints[-1].z - ground_z) > eps_z:
        distance_landing = abs(waypoints[-1].z - ground_z)
        travel_time += distance_landing / v_vert
        total_distance += distance_landing

    # ---- Turns per layer (count 90° corners) ----
    num_corners_total = 0
    turn_time_total = 0.0
    for li, vecs in layer_leg_vecs.items():
        n_corners = sum(1 for v1, v2 in zip(vecs, vecs[1:]) if is_right_angle(v1, v2))
        t_layer_turn = n_corners * t90
        layer_stats[li]['num_corners'] = n_corners
        layer_stats[li]['turn_time_90s'] = t_layer_turn
        num_corners_total += n_corners
        turn_time_total += t_layer_turn
    travel_time += turn_time_total

    # ---- Pauses (controller + fixed) ----
    controller_pause_time = controller_wp_pause_s * len(waypoints)
    fixed_overhead_time = fixed_pre_post_pauses_s

    # ---- Totals & flags ----
    total_time = travel_time + hover_time + controller_pause_time + fixed_overhead_time
    battery_used_pct = (total_time / drone.max_battery_time) * 100.0

    # Fill per-layer totals
    for ls in layer_stats:
        ls['total_time'] = ls['travel_time'] + ls['hover_time']
        ls['battery_usage_percent'] = (ls['total_time'] / drone.max_battery_time) * 100.0

    battery_ok = battery_used_pct <= 100.0
    battery_warning = battery_used_pct >= (drone.battery_warning_threshold * 100.0)
    distance_ok = total_distance <= drone.max_distance
    feasible = battery_ok and distance_ok

    metrics = {
        'distance': total_distance,
        'distance_takeoff': distance_takeoff,
        'distance_cruise': distance_cruise,
        'distance_landing': distance_landing,
        'travel_time': travel_time,  # includes turns + seams + verticals
        'hover_time': hover_time,
        'controller_pause_time': controller_pause_time,
        'fixed_overhead_time': fixed_overhead_time,
        'total_time': total_time,
        'battery_usage_percent': battery_used_pct,
        'battery_ok': battery_ok,
        'battery_warning': battery_warning,
        'distance_ok': distance_ok,
        'vertical_seams_count': vertical_seams_count,
        'vertical_seams_distance': vertical_seams_distance,
        'vertical_seams_time': vertical_seams_time,
        'num_corners': num_corners_total,
        't90_used_s': t90,
        'turn_time_90s': turn_time_total,
        'layers': layer_stats
    }
    return feasible, metrics



###### EXPORTING ######

def export_to_marvelmind(waypoints: List[Waypoint], filename: str):
    """
    export waypoints to csv file for marvelmind import
    columns: index, x, y, z, hold time, gimbal pitch
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'x', 'y', 'z', 'HoldTime', 'GimblePitch'])
        for idx, wp in enumerate(waypoints, start=1):
            writer.writerow([idx, wp.x, wp.y, wp.z, wp.hold_time, wp.gimbal_pitch])

def export_to_dpt(waypoints: List[Waypoint], drone_config: DroneConfig, filename: str):
    """
    exports waypoints and drone settings to a .dpt file format based on the provided example.
    This function makes assumptions about fixed command sequences and some setting values.
    """
    lines = []

    # --- Commands and Waypoints Section ---
    lines.append(f"takeoff({drone_config.min_altitude:.2f})")
    lines.append("pause(4.0)") # Based on your example
    lines.append(f"height({drone_config.min_altitude:.2f})")
    lines.append("pause(1.0)") # Based on your example
    lines.append("waypoints_begin()")

    for i, wp in enumerate(waypoints):
        # Waypoints in .dpt format are Wxx(x,y,z)
        lines.append(f"W{i+1:02d}({wp.x:.2f},{wp.y:.2f},{wp.z:.2f})")

    lines.append("waypoints_end()")
    lines.append("pause(1.0)") # Based on your example
    lines.append("landing()")
    lines.append("pause(6.0)") # Based on your example
    lines.append("landing()") # Second landing command from example

    # --- Settings Section ---
    lines.append("\n[settings]") # Add a newline for separation and the section header

    # These settings are based on the example .dpt file you provided.
    # You might want to make some of these configurable or derive them from DroneConfig.
    lines.append(f"set_power(10.00)")
    lines.append(f"set_power_rot(46.67)")
    lines.append(f"set_power_height(6.67)")
    lines.append(f"set_waypoint_pause(4.0)") # Fixed value as requested in prior turn
    lines.append(f"set_timeout(9.0)")
    lines.append(f"set_scan_timeout(10.0)")
    lines.append(f"set_waypoint_radius(0.05)")
    lines.append(f"set_wp1_radius_coef(5.0)")
    lines.append(f"set_waypoint_radius_z(0.10)")
    lines.append(f"set_recalibrate_distance(0.50)")
    lines.append(f"set_recalibrate_deviation(0.10)")
    lines.append(f"set_min_rotation_angle(10)")
    lines.append(f"set_angle_control_mode(0)")
    lines.append(f"set_pid_angle_distance(0.30)")
    lines.append(f"set_pid_angle_p(0.050)")
    lines.append(f"set_pid_angle_i(0.005)")
    lines.append(f"set_pid_angle_d(0.005)")
    lines.append(f"set_sliding_window(10)")
    lines.append(f"set_jump_sigma(0.100)")
    lines.append(f"set_no_tracking_fly_distance(1.50)")
    lines.append(f"set_no_tracking_a_param(0.900)")
    lines.append(f"set_no_tracking_c_param(0.100)")
    lines.append(f"set_no_tracking_time_coef(0.100)")
    lines.append(f"set_overflight_distance(0.080)")
    lines.append(f"set_overflight_samples(2)")
    lines.append(f"set_rotation_cor(0)")
    lines.append(f"set_min_rotation_cor(45)")
    lines.append(f"set_stop_spot_distance(8.0)")
    lines.append(f"set_stop_coef(0.80)")
    lines.append(f"set_video_recording(1)")
    lines.append(f"set_quality_hyst(30)")
    lines.append(f"set_time_limit_coef(1.30)")
    lines.append(f"set_reverse_brake_time(0.3)")
    lines.append(f"set_reverse_brake_power(20.0)")
    lines.append(f"set_reverse_brake_dist_a(0.3)")
    lines.append(f"set_reverse_brake_dist_b(0.1)")
    lines.append(f"set_rot_low_angle_speed(1.3)")
    lines.append(f"set_min_speed({drone_config.speed:.2f})") # Using drone's nominal speed
    lines.append(f"set_z_sliding_window(32)")
    lines.append(f"set_reverse_brake_time_c(0.67)")
    lines.append(f"set_reverse_brake_time_d(0.33)")

    # Write all lines to the .dpt file
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))





# --- MAIN SCRIPT ---
if __name__ == '__main__':
    cfg = DroneConfig(
        weight=0.292,
        max_battery_time=1380.0,
        horizontal_fov=82.1,
        vertical_fov=52.3,
        fps=60.0,
        resolution=(2720, 1530),
        speed=0.4,
        min_altitude=1.0,
        hover_buffer=10,
 