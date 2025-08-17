import math
import json
import base64
import pathlib
import datetime as dt

from dataclasses import dataclass, is_dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash.dependencies import ALL

import LogReportGenerator


# ========================================================================================================================
# ====================================        FLIGHTPATH GENERATOR DASHBOARD    ==========================================
# 
#  Dashboard for generating and visualising optimized flight paths for indoor drone scans. 
#  Users can set parameters through the UI, generate paths, export them in multiple formats (csv, dpt, txt), and analyse
#  the flight reports.
# 
#  Features:
#   - generate optimized flight path from user-defined inputs
#   - visualize 3d path interactively
#   - export results in different formats
#   - display summary statistics for the planned flight
#   - generate reports from uploaded drone log files
# ========================================================================================================================


#=== data structures ===
@dataclass
class DroneConfig:
    #basic drone properties and limits
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
    #single point in path. Includes position, gimbal pitch, speed and hold time.
    x: float
    y: float
    z: float
    gimbal_pitch: float
    speed: float
    hold_time: float


def calculate_footprint(drone: DroneConfig, distance: float) -> Tuple[float, float]:
    """
    computes camera ground footprint at a given range.

    args:
        drone: DroneConfig with horizontal_fov and vertical_fov in degrees
        distance: range from camera to target plane (m)
    returns:
        (width_m, height_m) footprint dimensions on the ground (m)
    """
    hfov = math.radians(drone.horizontal_fov)
    vfov = math.radians(drone.vertical_fov)
    w = 2 * distance * math.tan(hfov / 2)
    h = 2 * distance * math.tan(vfov / 2)
    return w, h


def calculate_speed(footprint_len: float, overlap: float, fps: float) -> float:
    """
    compute linear scan speed to achieve a desired along-track overlap

    args:
        footprint_len: along-track footprint length per frame (m)
        overlap: desired fractional overlap between frames [0, 1)
        fps: camera frame rate (frames/s)
    returns:
        linear scan speed (m/s)
    """
    #map fps and overlap to spatial step per second
    return footprint_len * (1 - overlap) * fps


#=== path generation: inward spiral on a single z ===

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
    generate an inward rectangular spiral at a fixed altitude
    the step size derives from optical footprint and desired overlap. the path
    respects a wall offset and floor clearance, starts at the center, then
    expands outward in axis-aligned legs.

    args:
        drone: platform parameters (fps, fov, speeds, buffers)
        dims: room dimensions (x_len, y_len, z_len) in meters
        altitude: flight altitude for this layer (m)
        overlap: fractional image overlap in [0, 1)
        wall_offset: extra margin beyond clearance along x walls (m)
        clearance: buffer from boundaries and floor (m)
        target_turns: soft cap for number of outward "rings"
    returns:
        list of Waypoint at the given altitude
    """
    #apply horizontal and vertical clearance from walls and floor
    inset_x = wall_offset + clearance
    inset_y = clearance
    #hold time at each waypoint is at least frame time plus buffer
    hold = 1.0 / drone.fps + drone.hover_buffer

    min_x = inset_x
    max_x = dims[0] - inset_x
    min_y = inset_y
    max_y = dims[1] - inset_y
    #invalid if buffer consumes full area
    if min_x >= max_x or min_y >= max_y:
        return []
    
    #enforce minimum flight altitude for safety purposes
    altitude = max(float(altitude), float(clearance or 0.0), float(drone.min_altitude))

    #determine scan step size from footprint
    fp_long, fp_short = calculate_footprint(drone, altitude)
    avg_fp = (fp_long + fp_short) / 2.0
    nominal_step = avg_fp * (1 - overlap)
    speed = calculate_speed(avg_fp, overlap, drone.fps)

    span_x = max_x - min_x
    span_y = max_y - min_y
    min_span = min(span_x, span_y)
    #reduce step to guarantee target number of turns fits inside area
    step = min(nominal_step, min_span / (2 * target_turns + 1))

    #start in center of area
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    x, y = cx, cy

    wps: List[Waypoint] = [Waypoint(x, y, altitude, 0.0, speed, hold)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    layer = 1
    dir_idx = 0

    #spiral loop: two segments per layer before expanding
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


#=== waypoint optimization: drop collinear points ===

def reduce_waypoints_collinear(waypoints: List[Waypoint], eps=1e-6) -> List[Waypoint]:
    """
    remove interior waypoints that are collinear and forward-progressing
    a waypoint is removed if three consecutive points lie on the same line
    (small cross magnitude) and the motion does not reverse (positive dot).

    args:
        waypoints: sequence of Waypoint
        eps: tolerance for cross magnitude to consider collinear
    returns:
        simplified list with endpoints preserved
    """
    #remove interior waypoints that do not change direction
    if len(waypoints) <= 2:
        return waypoints.copy()
    reduced = [waypoints[0]]
    for prev, curr, nxt in zip(waypoints, waypoints[1:], waypoints[2:]):
        v1 = (curr.x - prev.x, curr.y - prev.y, curr.z - prev.z)
        v2 = (nxt.x - curr.x, nxt.y - curr.y, nxt.z - curr.z)
        #cross product magnitude near zero means same line
        cross = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )
        cross_norm = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        #drop point if aligned and pointing forward
        if cross_norm < eps and dot > 0:
            continue
        reduced.append(curr)
    reduced.append(waypoints[-1])
    return reduced


#=== layering: reuse base spiral across multiple z levels for 3D scans, aim is to reduce multiple altitude values ===

def stacked_spiral(heights: Tuple[float], base_waypoints: List[Waypoint]) -> List[Waypoint]:
    """
    stack a 2d path across multiple altitudes, alternating direction
    this replicates the base path for each height in 'heights'. to reduce dead
    transit between layers, the order alternates (boustrophedon) based on the
    preceding endpoint.
    args:
        heights: tuple of z values (m) in the desired visiting order
        base_waypoints: 2d path (constant z) to reuse across layers
    returns:
        concatenated list of Waypoint with z set per layer
    """
    if not heights or not base_waypoints:
        return []

    def clone_with_z(seq: List[Waypoint], z: float) -> List[Waypoint]:
        return [Waypoint(w.x, w.y, z, w.gimbal_pitch, w.speed, w.hold_time) for w in seq]

    out: List[Waypoint] = []
    forward = base_waypoints
    reverse = list(reversed(base_waypoints))
    
    #first layer uses forward order
    current_layer = clone_with_z(forward, heights[0])
    out.extend(current_layer)

    base_start_xy = (forward[0].x, forward[0].y)
    base_end_xy = (forward[-1].x, forward[-1].y)

    #subsequent layers alternate orientation based on previous end point 
    #(first point of subsequent layer is last point of previous layer, altitude changes)
    for z in heights[1:]:
        prev_last_xy = (out[-1].x, out[-1].y)
        if prev_last_xy == base_end_xy:
            oriented = reverse
        elif prev_last_xy == base_start_xy:
            oriented = forward
        else:
            oriented = reverse
        out.extend(clone_with_z(oriented, z))

    return out


#=== mission validation: global metrics without layer detail ===

def validate_mission(
    drone: DroneConfig,
    waypoints: List[Waypoint],
    include_takeoff_landing: bool = True,
    ground_z: float = 0.0,
    controller_wp_pause_s: float = 2.0,
    fixed_pre_post_pauses_s: float = 12.0,
) -> Tuple[bool, dict]:
    """
    evaluate mission feasibility and aggregate metrics (single summary)
    counts path distance, time, and simple yaw-turn overhead using a coarse
    yaw-rate model. optionally includes vertical takeoff/landing segments.

    args:
        drone: platform parameters and limits
        waypoints: full 3d path to evaluate
        include_takeoff_landing: whether to add vertical segments to ground_z
        ground_z: ground reference altitude (m)
        controller_wp_pause_s: per-waypoint controller pause (s)
        fixed_pre_post_pauses_s: fixed overhead before/after mission (s)
    returns:
        (feasible, metrics) where feasible considers battery and distance caps
    """
    if not waypoints:
        return False, {
            "distance": 0.0,
            "distance_takeoff": 0.0,
            "distance_cruise": 0.0,
            "distance_landing": 0.0,
            "num_corners": 0,
            "t90_used_s": 0.0,
            "turn_time_90s": 0.0,
            "travel_time": 0.0,
            "hover_time": 0.0,
            "controller_pause_time": 0.0,
            "fixed_overhead_time": 0.0,
            "total_time": 0.0,
            "battery_usage_percent": 0.0,
            "battery_ok": True,
            "battery_warning": False,
            "distance_ok": True,
        }

    v_horiz = drone.speed
    v_vert = getattr(drone, "vertical_speed", None) or drone.speed
    #estimate yaw rate and time to complete 90° turns (parameters from marvelmind specs)
    base_power = 60.0
    base_yaw_deg_s = 120.0
    power_rot = 46.67
    yaw_rate_large = base_yaw_deg_s * (power_rot / base_power)
    t90 = 90.0 / max(yaw_rate_large, 1e-6)

    total_distance = 0.0
    travel_time = 0.0

    #takeoff climb if starting above ground
    distance_takeoff = 0.0
    if include_takeoff_landing and waypoints[0].z != ground_z:
        distance_takeoff = abs(waypoints[0].z - ground_z)
        travel_time += distance_takeoff / v_vert
        total_distance += distance_takeoff

    #iterate through cruise legs
    distance_cruise = 0.0
    leg_vecs = []
    for wp1, wp2 in zip(waypoints, waypoints[1:]):
        dx, dy = wp2.x - wp1.x, wp2.y - wp1.y
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            continue
        d2 = math.hypot(dx, dy)
        distance_cruise += d2
        total_distance += d2
        travel_time += d2 / max(v_horiz, 1e-9)
        leg_vecs.append((dx, dy))

    def is_right_angle(v1, v2, tol_deg=15.0):
        """
        return true if the angle between v1 and v2 is approximately 90 degrees

        args:
            v1, v2: 2d vectors (dx, dy)
            tol_deg: angular tolerance around 90° (deg)
        returns:
            bool indicating right-angle corner
        """
        #corner detection with tolerance around 90°
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return False
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.degrees(math.acos(cosang))
        return (90.0 - tol_deg) <= ang <= (90.0 + tol_deg)
    
    #count corners and add turn time
    num_corners = sum(1 for v1, v2 in zip(leg_vecs, leg_vecs[1:]) if is_right_angle(v1, v2))
    turn_time = num_corners * t90
    travel_time += turn_time

    #landing descent if ending above ground
    distance_landing = 0.0
    if include_takeoff_landing and waypoints[-1].z != ground_z:
        distance_landing = abs(waypoints[-1].z - ground_z)
        travel_time += distance_landing / v_vert
        total_distance += distance_landing

    #additional timing components
    hover_time = sum(getattr(wp, "hold_time", 0.0) for wp in waypoints)
    controller_pause_time = controller_wp_pause_s * len(waypoints)
    fixed_overhead_time = fixed_pre_post_pauses_s

    #total time required calculation and battery usage from total mission time
    total_time = travel_time + hover_time + controller_pause_time + fixed_overhead_time
    battery_used_pct = (total_time / drone.max_battery_time) * 100.0

    battery_ok = battery_used_pct <= 100.0
    battery_warning = battery_used_pct >= (drone.battery_warning_threshold * 100.0)
    distance_ok = total_distance <= drone.max_distance
    #final assessment of feasibility
    feasible = battery_ok and distance_ok

    return feasible, {
        "distance": total_distance,
        "distance_takeoff": distance_takeoff,
        "distance_cruise": distance_cruise,
        "distance_landing": distance_landing,
        "num_corners": num_corners,
        "t90_used_s": t90,
        "turn_time_90s": turn_time,
        "travel_time": travel_time,
        "hover_time": hover_time,
        "controller_pause_time": controller_pause_time,
        "fixed_overhead_time": fixed_overhead_time,
        "total_time": total_time,
        "battery_usage_percent": battery_used_pct,
        "battery_ok": battery_ok,
        "battery_warning": battery_warning,
        "distance_ok": distance_ok,
    }


#=== mission validation with layers: per-z accounting ===

def validate_mission_with_layers(
    drone: DroneConfig,
    waypoints: List[Waypoint],
    include_takeoff_landing: bool = True,
    ground_z: float = 0.0,
    controller_wp_pause_s: float = 2.0,
    fixed_pre_post_pauses_s: float = 12.0,
) -> Tuple[bool, dict]:
    """
    evaluate feasibility with a per-layer breakdown and seam costs
    splits the path into contiguous constant-z layers, accumulates in-layer
    distance/time, counts per-layer corners, and charges mixed 2d+vertical
    moves as vertical seams between layers.

    args:
        drone: platform parameters and limits
        waypoints: full 3d path
        include_takeoff_landing: whether to add vertical segments to ground_z
        ground_z: ground reference altitude (m)
        controller_wp_pause_s: per-waypoint controller pause (s)
        fixed_pre_post_pauses_s: fixed overhead before/after mission (s)
    returns:
        (feasible, metrics) with metrics['layers'] holding per-layer dicts
    """
    if not waypoints:
        empty = {
            "distance": 0.0,
            "distance_takeoff": 0.0,
            "distance_cruise": 0.0,
            "distance_landing": 0.0,
            "travel_time": 0.0,
            "hover_time": 0.0,
            "controller_pause_time": 0.0,
            "fixed_overhead_time": 0.0,
            "total_time": 0.0,
            "battery_usage_percent": 0.0,
            "battery_ok": True,
            "battery_warning": False,
            "distance_ok": True,
            "vertical_seams_count": 0,
            "vertical_seams_distance": 0.0,
            "vertical_seams_time": 0.0,
            "num_corners": 0,
            "t90_used_s": 0.0,
            "turn_time_90s": 0.0,
            "layers": [],
        }
        return False, empty

    v_horiz = drone.speed
    v_vert = getattr(drone, "vertical_speed", None) or drone.speed
    eps_z = 1e-9

    #yaw rate estimate (from marvelmind specs)
    base_power = 60.0
    base_yaw_deg_s = 120.0
    power_rot = 46.67
    yaw_rate_large = base_yaw_deg_s * (power_rot / base_power)
    t90 = 90.0 / max(yaw_rate_large, 1e-6)

    #identify contiguous segments of constant z
    layers: List[Tuple[float, int, int]] = []
    start = 0
    current_z = waypoints[0].z
    for i in range(1, len(waypoints)):
        if abs(waypoints[i].z - current_z) > eps_z:
            layers.append((current_z, start, i - 1))
            start = i
            current_z = waypoints[i].z
    layers.append((current_z, start, len(waypoints) - 1))

    layer_stats = [
        {
            "z": z,
            "idx_start": s,
            "idx_end": e,
            "num_waypoints": (e - s + 1),
            "distance_2d": 0.0,
            "distance_3d": 0.0,
            "travel_time": 0.0,
            "hover_time": sum(getattr(wp, "hold_time", 0.0) for wp in waypoints[s : e + 1]),
            "num_corners": 0,
            "turn_time_90s": 0.0,
            "total_time": 0.0,
            "battery_usage_percent": 0.0,
        }
        for (z, s, e) in layers
    ]

    def which_layer(idx: int) -> int:
        """
        map an edge index (wp_i -> wp_{i+1}) to its layer index

        returns:
            layer index or -1 if not found
        """
        for li, (_, s, e) in enumerate(layers):
            if s <= idx <= e:
                return li
        return -1

    def is_right_angle(v1, v2, tol_deg=15.0) -> bool:
        """
        return true if the angle between v1 and v2 is approximately 90 degrees
        """
        #corner detection reused here for per-layer turns
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-12 or n2 < 1e-12:
            return False
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.degrees(math.acos(cosang))
        return (90.0 - tol_deg) <= ang <= (90.0 + tol_deg)

    total_distance = 0.0
    distance_takeoff = 0.0
    distance_cruise = 0.0
    distance_landing = 0.0

    travel_time = 0.0
    hover_time = sum(getattr(wp, "hold_time", 0.0) for wp in waypoints)

    #takeoff climb
    if include_takeoff_landing and abs(waypoints[0].z - ground_z) > eps_z:
        distance_takeoff = abs(waypoints[0].z - ground_z)
        travel_time += distance_takeoff / v_vert
        total_distance += distance_takeoff

    vertical_seams_count = 0
    vertical_seams_distance = 0.0
    vertical_seams_time = 0.0

    layer_leg_vecs: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(layers))}

    for i, (wp1, wp2) in enumerate(zip(waypoints, waypoints[1:])):
        dx, dy, dz = wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z
        d2 = math.hypot(dx, dy)
        d3 = math.sqrt(dx * dx + dy * dy + dz * dz)

        li1 = which_layer(i)
        li2 = which_layer(i + 1)

        total_distance += d3

        if li1 == li2 and li1 != -1 and abs(dz) <= eps_z:
            #in-layer horizontal motion
            distance_cruise += d2
            layer_stats[li1]["distance_2d"] += d2
            layer_stats[li1]["distance_3d"] += d3
            t_seg = d2 / max(v_horiz, 1e-9)
            layer_stats[li1]["travel_time"] += t_seg
            travel_time += t_seg
            if d2 > 1e-12:
                layer_leg_vecs[li1].append((dx, dy))
        else:
            #vertical seam or mixed move between layers
            vertical_seams_count += 1
            vertical_seams_distance += d3
            distance_cruise += d3
            t_seam = 0.0
            if d2 > 0.0:
                t_seam += d2 / max(v_horiz, 1e-9)
            if abs(dz) > 0.0:
                t_seam += abs(dz) / max(v_vert, 1e-9)
            vertical_seams_time += t_seam
            travel_time += t_seam
    #landing descent
    if include_takeoff_landing and abs(waypoints[-1].z - ground_z) > eps_z:
        distance_landing = abs(waypoints[-1].z - ground_z)
        travel_time += distance_landing / v_vert
        total_distance += distance_landing

    num_corners_total = 0
    turn_time_total = 0.0
    for li, vecs in layer_leg_vecs.items():
        n_corners = sum(1 for v1, v2 in zip(vecs, vecs[1:]) if is_right_angle(v1, v2))
        t_layer_turn = n_corners * t90
        layer_stats[li]["num_corners"] = n_corners
        layer_stats[li]["turn_time_90s"] = t_layer_turn
        num_corners_total += n_corners
        turn_time_total += t_layer_turn
    travel_time += turn_time_total

    controller_pause_time = controller_wp_pause_s * len(waypoints)
    fixed_overhead_time = fixed_pre_post_pauses_s

    total_time = travel_time + hover_time + controller_pause_time + fixed_overhead_time
    battery_used_pct = (total_time / drone.max_battery_time) * 100.0

    #finalise per layer totals and normalized battery usage
    for ls in layer_stats:
        ls["total_time"] = ls["travel_time"] + ls["hover_time"]
        ls["battery_usage_percent"] = (ls["total_time"] / drone.max_battery_time) * 100.0

    battery_ok = battery_used_pct <= 100.0
    battery_warning = battery_used_pct >= (drone.battery_warning_threshold * 100.0)
    distance_ok = battery_ok and ((drone.max_distance is None) or (total_distance <= float(drone.max_distance)))
    feasible = battery_ok and distance_ok

    metrics = {
        "distance": total_distance,
        "distance_takeoff": distance_takeoff,
        "distance_cruise": distance_cruise,
        "distance_landing": distance_landing,
        "travel_time": travel_time,
        "hover_time": hover_time,
        "controller_pause_time": controller_pause_time,
        "fixed_overhead_time": fixed_overhead_time,
        "total_time": total_time,
        "battery_usage_percent": battery_used_pct,
        "battery_ok": battery_ok,
        "battery_warning": battery_warning,
        "distance_ok": distance_ok,
        "vertical_seams_count": vertical_seams_count,
        "vertical_seams_distance": vertical_seams_distance,
        "vertical_seams_time": vertical_seams_time,
        "num_corners": num_corners_total,
        "t90_used_s": t90,
        "turn_time_90s": turn_time_total,
        "layers": layer_stats,
    }
    return feasible, metrics

#===========================================================
# Waypoint export functions
#===========================================================

def render_marvelmind_csv(waypoints: List[Waypoint]) -> str:
    """
    convert a list of waypoints into marvelmind csv format
    fields: index, x, y, z, hold_time, gimbal_pitch
    returns a string ready to be written as a csv file
    """
    # index, x, y, z, HoldTime, GimblePitch
    lines = ["index,x,y,z,HoldTime,GimblePitch"]
    for idx, wp in enumerate(waypoints, start=1):
        lines.append(f"{idx},{wp.x:.6f},{wp.y:.6f},{wp.z:.6f},{wp.hold_time:.3f},{wp.gimbal_pitch:.2f}")
    return "\n".join(lines)


def render_dpt(waypoints: List[Waypoint], drone_config: DroneConfig) -> str:
    """
    convert a list of waypoints and a drone configuration into dpt script format
    includes takeoff, pauses, waypoint list, landing, and drone settings
    returns a string ready to be written into a .dpt file
    """
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
    """
    convert a list of waypoints into plain text format
    fields: index, x, y, z, hold_time, gimbal_pitch, speed
    returns a string ready to be written into a .txt file
    """
    lines = ["# index x y z hold_time gimbal_pitch speed"]
    for i, wp in enumerate(waypoints, start=1):
        lines.append(f"{i} {wp.x:.6f} {wp.y:.6f} {wp.z:.6f} {wp.hold_time:.3f} {wp.gimbal_pitch:.2f} {wp.speed:.3f}")
    return "\n".join(lines)








# ============================================================
# ========================  DASH APP  ========================
# ============================================================


#=== app init and static asssets ===

app = Dash(__name__, suppress_callback_exceptions=True, title="Fís - Indoor Drone Path Generator")
server = app.server  # for Gunicorn / PaaS
server.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # 64 MB
logo_path = pathlib.Path("/Users/stefaniaconte/Desktop/FlightGenerator_App/ucd_logo.jpg")
app_logo_path =pathlib.Path("/Users/stefaniaconte/Desktop/FlightGenerator_App/app_logo.png")
try:
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
    logo_src = f"data:image/jpeg;base64,{logo_b64}"
    app_logo_b64 = base64.b64encode(app_logo_path.read_bytes()).decode()
    app_logo_src = f"data:image/jpeg;base64,{app_logo_b64}"
except Exception:
    #fallback if logo not found; downstream ui should tolerate None
    logo_src = None
    app_logo_b64 = None


#=== components IDs (source of truth for callbacks) ===

IDS = dict(
    #app controls
    tabs="tabs",
    graph="graph",
    metrics="metrics",
    slider="slider",
    play="play",
    pause="pause",
    tick="tick",
    store_wps="store-wps",
    store_metrics="store-metrics",
    store_cfg="store-cfg",
    btn_generate="btn-generate",
    btn_reset="btn-reset",
    dl_csv="dl-csv",
    dl_txt="dl-txt",
    dl_dpt="dl-dpt",
    error="error",

    #home page “Generate Flight Report” 
    btn_report="btn-report",
    btn_report_link="btn-report-link",

    #report page IDs
    upload_log="upload-log",
    upload_path="upload-path",              
    btn_run_report="btn-run-report",        
    store_report_bundle="store-report-bundle",  
    report_output="report-output",         
)


#=== defaults and initial figures ====

DEFAULT_DRONE = DroneConfig(
    weight=0.292,
    max_battery_time=1380.0,
    max_distance=140,  
    horizontal_fov=82.1,
    vertical_fov=52.3,
    fps=60.0,
    resolution=(2720, 1530),
    speed=0.4,
    min_altitude=1.0,
    hover_buffer=5.0,
    battery_warning_threshold=0.8,
)

DEFAULT_ROOM = dict(length=7.2, width=3.8, height=3.1)

#pre-seed a minimal 2d room outline, 3d path will be layered on top 
INITIAL_FIG = go.Figure()
INITIAL_FIG.add_shape(type="rect",
                      x0=0, y0=0,
                      x1=DEFAULT_ROOM["length"], y1=DEFAULT_ROOM["width"],
                      line=dict(dash="dot"))
INITIAL_FIG.update_xaxes(range=[0, DEFAULT_ROOM["length"]], autorange=False, fixedrange=True, title_text="x (m)")
INITIAL_FIG.update_yaxes(range=[0, DEFAULT_ROOM["width"]],  autorange=False, fixedrange=True,
                         scaleanchor="x", scaleratio=1, title_text="y (m)")
INITIAL_FIG.update_layout(height=600, margin=dict(l=40, r=20, t=20, b=40))


#=== CSS styles for the app ===

TITLE_STYLE = {
    "margin": "0 0 8px 0",   
    "fontSize": "20px",
    "fontWeight": "600",
    "letterSpacing": "0.2px",
}
TIP_ICON_STYLE = {"cursor": "help", "color": "#888", "marginLeft": "6px", "fontWeight": 600}

REPORT_BTN_STYLE = {
    "background": "#28a745",
    "color": "white",
    "padding": "12px 20px",
    "fontSize": "16px",
    "fontWeight": "600",
    "border": "none",
    "borderRadius": "12px",
    "cursor": "pointer",
    "boxShadow": "0 2px 6px rgba(16,185,129,0.4)",
    "minWidth": "240px",
}

HOME_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "minmax(380px, 460px) 1fr",
    "gap": "16px",
    "padding": "16px",
    "alignItems": "start",
}

REPORT_STYLE = {"display": "block", "padding": "16px"}
HIDDEN = {"display": "none"}


#===========================================================
# Report generator layout
#===========================================================

def report_page_layout():
    """
    build the report page layout
    returns a dash html.Div containing:
      - two upload widgets (log csv and planned path csv)
      - a run button that enables when both files are present
      - a store for the report bundle
      - an output container for rendered report content
    """
    return html.Div(
        id="report-page",
        style={"display": "none", "padding": "16px"},
        children=[
            dcc.Store(id=IDS["store_report_bundle"]),
            html.H2("Generate Flight Report", style={"marginTop": 0}),
            html.P("Upload the raw Marvelmind log (.csv) and the planned flightpath (.csv), then click Run Report."),

            #uploads
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(260px, 1fr))",
                    "gap": "12px",
                    "alignItems": "stretch",
                    "margin": "12px 0 8px",
                },
                children=[
                    dcc.Upload(
                        id=IDS["upload_log"],
                        children=html.Div([
                            html.Strong("Log file (.csv)"),
                            html.Div("Drag & drop or click to upload", style={"color": "#666", "fontSize": "12px"})
                        ]),
                        multiple=False,
                        accept=".csv",
                        style={
                            "border": "2px dashed #d1d5db",
                            "borderRadius": "12px",
                            "padding": "16px",
                            "textAlign": "center",
                            "cursor": "pointer",
                            "background": "#fafafa",
                        },
                    ),
                    dcc.Upload(
                        id=IDS["upload_path"],
                        children=html.Div([
                            html.Strong("Flightpath (.csv)"),
                            html.Div("Drag & drop or click to upload", style={"color": "#666", "fontSize": "12px"})
                        ]),
                        multiple=False,
                        accept=".csv",
                        style={
                            "border": "2px dashed #d1d5db",
                            "borderRadius": "12px",
                            "padding": "16px",
                            "textAlign": "center",
                            "cursor": "pointer",
                            "background": "#fafafa",
                        },
                    ),
                ],
            ),

            #run button (enabled only when both uploads present)
            html.Div(
                style={"display": "flex", "justifyContent": "center", "margin": "12px 0 20px"},
                children=[
                    html.Button(
                        "Run report",
                        id=IDS["btn_run_report"],
                        n_clicks=0,
                        disabled=True,
                        style={
                            "padding": "12px 18px",
                            "background": "#10b981",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "12px",
                            "fontWeight": 700,
                            "fontSize": "16px",
                            "cursor": "not-allowed",
                            "opacity": 0.5,
                            "boxShadow": "0 2px 6px rgba(16,185,129,0.3)",
                        },
                        title="Upload both files to enable",
                    ),
                ],
            ),

            #output area: markdown + images (collapsible)
            html.Div(id=IDS["report_output"], style={"maxWidth": "980px", "margin": "0 auto"}),
        ],
    )


#---- Layout ----
app.layout = html.Div(
    className="app",
    children=[
        dcc.Location(id="url", refresh=False), 
        dcc.Store(id=IDS["store_wps"]),
        dcc.Store(id=IDS["store_metrics"]),
        dcc.Store(id=IDS["store_cfg"], data={"scan_mode": "2D"}),
        dcc.Interval(id=IDS["tick"], interval=500, disabled=True),

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
                html.Div([
                    html.H1(
                        "Fís - Indoor Drone Path Generator",
                        style={"margin": 0, "fontSize": "35px", "letterSpacing": "0.3px"}
                    ),
                    html.Div([
                        html.Div("One of the first platforms on the market to provide a streamlined solution for indoor 3D scanning and reconstruction.", 
                                style={"marginBottom": "6px"}),
                        html.Div("Welcome to Fìs. Enjoy planning, flying, and reconstructing!"),
                    ],
                    style={"color": "#666", "fontSize": "20px", "maxWidth": "1000px"}

                    ),
                ]),

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

        html.Div(
            id = "home-page",
            className="container",
            style={"display": "grid", "gridTemplateColumns": "minmax(380px, 460px) 1fr", "gap": "16px", "padding": "16px", "alignItems": "start"},
            children=[
                # ----- Sidebar -----
                html.Div(
                    className="sidebar",
                    style={"display": "flex", "flexDirection": "column", "gap": "8px"},
                    children=[
                        html.H3("Controls", style=TITLE_STYLE),
                        html.Label("Room dimensions (m)"),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)", "gap": "8px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                    children=[
                                        html.Label("Length L (m)", htmlFor="length", style={"fontSize": 12, "color": "#555"}),
                                        dcc.Input(id="length", type="number", value=DEFAULT_ROOM["length"], min=0.5, step=0.1, placeholder="Length"),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                    children=[
                                        html.Label("Width W (m)", htmlFor="width", style={"fontSize": 12, "color": "#555"}),
                                        dcc.Input(id="width", type="number", value=DEFAULT_ROOM["width"], min=0.5, step=0.1, placeholder="Width"),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                    children=[
                                        html.Label("Height H (m)", htmlFor="height", style={"fontSize": 12, "color": "#555"}),
                                        dcc.Input(id="height", type="number", value=DEFAULT_ROOM["height"], min=0.5, step=0.1, placeholder="Height"),
                                    ],
                                ),
                            ],
                        )
                        ,
                        html.Label("Scan mode"),
                        dcc.Dropdown(
                            id="scan-mode",
                            options=[{"label": "2D", "value": "2D"}, {"label": "3D", "value": "3D"}],
                            value="2D",
                            clearable=False,
                        ),
                        html.Hr(),
                        #--- spiral/camera parameters ---
                    html.H3("Parameter Setup", style=TITLE_STYLE),

                    #overlap slider (avoid cluttering)
                    dcc.Slider(
                        id="overlap",
                        min=0.0, max=0.9, step=0.05, value=0.3,
                        marks={0.0: "0.0", 0.3: "0.3", 0.6: "0.6", 0.9: "0.9"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Div("Overlap (0–0.9)", style={"fontSize": 12, "color": "#666"}),

                    #labels
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(2,1fr)", "gap": "10px", "marginTop": "6px"},
                        children=[
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                children=[
                                    html.Label([
                                    "Wall offset (m)",
                                    html.Span(" ⓘ", title="Minimum distance kept from the room walls.", style=TIP_ICON_STYLE),
                                ], htmlFor="wall-offset", style={"fontSize": 12, "color": "#555"}),
                                    dcc.Input(id="wall-offset", type="number", value=0.2, min=0.0, step=0.05, placeholder="e.g., 0.2"),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                children=[
                                    html.Label([
                                    "Safety clearance (m)",
                                    html.Span(" ⓘ", title="Extra buffer from obstacles/furniture.", style=TIP_ICON_STYLE),
                                ], htmlFor="clearance", style={"fontSize": 12, "color": "#555"}),
                                    dcc.Input(id="clearance", type="number", value=0.2, min=0.0, step=0.05, placeholder="e.g., 0.2"),
                                ],
                            ),
                        ],
                    ),

                    #--- spiral basics (labels + tooltips) ---
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(2,1fr)", "gap": "10px", "marginTop": "6px"},
                        children=[
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                children=[
                                    html.Label(
                                        [
                                            "Altitude z (m)",
                                            html.Span(
                                                " ⓘ",
                                                title="Altitude above the floor at which this spiral is flown.",
                                                style=TIP_ICON_STYLE,
                                            ),
                                        ],
                                        htmlFor="altitude",
                                        style={"fontSize": 12, "color": "#555"},
                                    ),
                                    dcc.Input(
                                        id="altitude", type="number", value=1.2, min=0.1, step=0.1,
                                        placeholder="e.g., 1.2",
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                children=[
                                    html.Label(
                                        [
                                            "Target turns in spiral",
                                            html.Span(
                                                " ⓘ",
                                                title="Number of 90° turns/rings; higher = tighter coverage and longer flight time.",
                                                style=TIP_ICON_STYLE,
                                            ),
                                        ],
                                        htmlFor="turns",
                                        style={"fontSize": 12, "color": "#555"},
                                    ),
                                    dcc.Input(
                                        id="turns", type="number", value=3, min=1, step=1,
                                        placeholder="e.g., 3",
                                    ),
                                ],
                            ),
                        ],
                    ),

                    #3D-only fields (hidden unless scan-mode == "3D")
                    html.Div(
                        id="3d-only",
                        style={"display": "none", "marginTop": "6px"},
                        children=[
                            #--- fixed 3-column row ---
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)", "gap": "10px"},
                                children=[
                                    #number of spiral layers
                                    html.Div(
                                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                        children=[
                                            html.Label(
                                                [
                                                    "Nr. Spiral layers",
                                                    html.Span(" ⓘ", title="How many distinct altitudes (layers) to fly.", style=TIP_ICON_STYLE),
                                                ],
                                                htmlFor="layers",
                                                style={"fontSize": 12, "color": "#555"},
                                            ),
                                            dcc.Input(id="layers", type="number", value=3, min=1, step=1, placeholder="e.g., 3"),
                                        ],
                                    ),

                                    #optional top cap
                                    html.Div(
                                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                        children=[
                                            html.Label(
                                                [
                                                    "Z limit (m, optional)",
                                                    html.Span(
                                                        " ⓘ",
                                                        title="Maximum altitude for the stacked layers. Leave blank to include all computed heights.",
                                                        style=TIP_ICON_STYLE,
                                                    ),
                                                ],
                                                htmlFor="top-cap",
                                                style={"fontSize": 12, "color": "#555"},
                                            ),
                                            dcc.Input(id="top-cap", type="number", placeholder="e.g., 2.4"),
                                        ],
                                    ),
                                ],
                            ),

                            #--- dynamic per-gap Δz inputs (one per gap) ---
                            html.Div(id="dz-container", style={"marginTop": "8px"}, children=[]),
                        ],
                    ),
                        

                        html.Hr(),

                        html.Hr(),
                        html.Div(
                            style={"display": "flex", "gap": "8px"},
                            children=[
                                html.Button(
                                    "Generate Path",
                                    id=IDS["btn_generate"],
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#28a745",   # green
                                        "color": "white",
                                        "padding": "12px 20px",
                                        "width": "300px",   
                                        "border": "none",
                                        "borderRadius": "12px",
                                        "fontSize": "16px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Reset",
                                    id=IDS["btn_reset"],
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#dc3545",   # red
                                        "color": "white",
                                        "padding": "10px 18px",
                                        "width": "300px",   
                                        "border": "none",
                                        "borderRadius": "6px",
                                        "fontSize": "16px",
                                        "cursor": "pointer",
                                        "marginLeft": "10px",
                                    },
                                ),
                            ],
                        ),
                        html.Div(id=IDS["error"], style={"color": "crimson", "fontSize": 13}),
                        html.Hr(),

                        #--- mission summary card (filled by callback) ---
                        html.Div(
                            id="metrics-card",
                            style={
                                "background": "#f7f9ff",
                                "border": "1px solid #dfe7ff",
                                "padding": "12px 14px",
                                "borderRadius": "10px",
                                "margin": "8px 0 12px 0",
                                "lineHeight": "1.35",
                            },
                            children=[
                                html.Strong("Mission summary"),
                                html.Div("Click “Generate Path” to see metrics.", style={"color": "#666", "marginTop": "6px"}),
                            ],
                        ),

                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)", "gap": "6px"},
                            children=[
                                html.Button("Download CSV", id="btn-csv"),
                                html.Button("Download TXT", id="btn-txt"),
                                html.Button("Download DPT", id="btn-dpt"),
                            ],
                        ),
                        dcc.Download(id=IDS["dl_csv"]),
                        dcc.Download(id=IDS["dl_txt"]),
                        dcc.Download(id=IDS["dl_dpt"]),
                    ],
                ),

                #----- main panel -----
                html.Div(
                    className="main",
                    style={"minWidth": 0, "position": "relative", "overflow": "hidden"},
                    children=[
                        html.Div(
                            id="panel-wrap",
                            style={"width": "100%", "maxWidth": "1500px", "marginLeft": "auto"},
                            children=[
                                html.H3(
                                    "Path View",
                                     style={**TITLE_STYLE, "borderBottom": "1px solid #eaeaea", "paddingBottom": "6px"},
                                ),
                                html.Div(
                                    id="panel",
                                    children=[
                                        dcc.Graph(
                                            id=IDS["graph"],
                                            figure=INITIAL_FIG,  
                                            style={"height": "75vh", "width": "100%"},
                                        ),
                                        html.Div(
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "auto 1fr auto auto auto",
                                                "alignItems": "center",
                                                "gap": "4px",
                                            },
                                            children=[
                                                html.Div("Step:"),
                                                dcc.Slider(
                                                    id=IDS["slider"],
                                                    min=0,
                                                    max=0,
                                                    step=1,
                                                    value=0,
                                                    marks={},
                                                    tooltip={"placement": "bottom"},
                                                ),
                                                html.Button("Play", id=IDS["play"], n_clicks=0),
                                                html.Button("Pause", id=IDS["pause"], n_clicks=0),
                                            ],
                                        ),
                                        html.Div(
                                            id="report-controls",
                                            style={"display": "flex", "justifyContent": "center", "marginTop": "12px"},
                                            children=[
                                                dcc.Link(
                                            html.Button("Generate Flight Report", id=IDS["btn_report"], n_clicks=0, style=REPORT_BTN_STYLE),
                                            href="/report",
                                            id=IDS["btn_report_link"],
                                            refresh=False,
                                        ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    #separate route for generating a full flight report
    report_page_layout(),
    ],
)    


# ==============================================================
# Helper functions 
# ==============================================================

def wps_to_df(wps: List[Waypoint]) -> pd.DataFrame:
    """
    convert waypoints to a dataframe ready for plotting and summaries

    args:
        wps: list of Waypoint
    returns:
        pd.DataFrame with columns: x,y,z,hold,gimbal,speed,step,cum_dist_m
    """
    if not wps:
        return pd.DataFrame(columns=["x", "y", "z", "hold", "gimbal", "speed", "step"]).astype({"step": int})
    df = pd.DataFrame(
        {
            "x": [w.x for w in wps],
            "y": [w.y for w in wps],
            "z": [w.z for w in wps],
            "hold": [w.hold_time for w in wps],
            "gimbal": [w.gimbal_pitch for w in wps],
            "speed": [w.speed for w in wps],
        }
    )
    df["step"] = np.arange(len(df))
    # cumulative distance for charts
    p = df[["x", "y", "z"]].to_numpy()
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.array([])
    cum = np.concatenate([[0.0], np.cumsum(seg)]) if len(p) > 0 else np.array([0.0])
    df["cum_dist_m"] = cum
    return df

def make_2d_fig(df: pd.DataFrame, L: float, W: float, step_idx: int) -> go.Figure:
    """
    create a 2d plan-view figure of the path with current position

    args:
        df: dataframe from wps_to_df
        L: room length (m)
        W: room width (m)
        step_idx: current step index for the playback slider
    returns:
        plotly Figure configured for 2d view
    """
    fig = go.Figure()

    if not df.empty:
        show = df[df["step"] <= step_idx]
        fig.add_trace(go.Scatter(x=show["x"], y=show["y"], mode="lines+markers", name="path"))
        fig.add_trace(go.Scatter(x=[df.loc[0, "x"]], y=[df.loc[0, "y"]],
                                 mode="markers", name="start", marker_symbol="star"))
        fig.add_trace(go.Scatter(x=[df.loc[len(df)-1, "x"]], y=[df.loc[len(df)-1, "y"]],
                                 mode="markers", name="end", marker_symbol="x"))
        cur = df.iloc[min(step_idx, len(df)-1)]
        fig.add_trace(go.Scatter(x=[cur.x], y=[cur.y], mode="markers",
                                 name="current", marker_size=12))

    #lock ranges to start at (0,0) and do not square the aspect (plotly defaults to square)
    fig.update_xaxes(
        range=[0, L],
        autorange=False,
        fixedrange=True,
        constrain="range",
        tick0=0, dtick=0.5,
        zeroline=False,
        showline=True, mirror=True, linecolor="#444", linewidth=1,
        ticks="outside",
        title_text="x (m)"
    )
    fig.update_yaxes(
        range=[0, W],
        autorange=False,
        fixedrange=True,
        constrain="range",
        tick0=0, dtick=0.5,
        zeroline=False,
        showline=True, mirror=True, linecolor="#444", linewidth=1,
        ticks="outside",
        title_text="y (m)"
    )

    fig.update_layout(
        template="plotly",
        height=800,
        margin=dict(l=40, r=20, t=20, b=16)  
    )
    return fig


def make_3d_fig(df: pd.DataFrame, L: float, W: float, H: float, step_idx: int) -> go.Figure:
    """
    create a 3d figure of the path with current position

    args:
        df: dataframe from wps_to_df
        L, W, H: room dimensions (m)
        step_idx: current step index for the playback slider
    returns:
        plotly Figure configured for 3d view
    """
    fig = go.Figure()

    if not df.empty:
        show = df[df["step"] <= step_idx]
        fig.add_trace(go.Scatter3d(
            x=show["x"], y=show["y"], z=show["z"],
            mode="lines+markers", name="path"
        ))
        cur = df[df["step"] == min(step_idx, len(df)-1)].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=[cur.x], y=[cur.y], z=[cur.z],
            mode="markers", name="current", marker_size=5
        ))

    zoom_out_factor = 0.8  #avoid zooming in
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m)", range=[0, L], autorange=False),
            yaxis=dict(title="y (m)", range=[0, W], autorange=False),
            zaxis=dict(title="z (m)", range=[0, H], autorange=False),
            aspectmode="manual",
            aspectratio=dict(x=L, y=W, z=H),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(
                    x=zoom_out_factor * max(L, W, H),
                    y=zoom_out_factor * max(L, W, H),
                    z=zoom_out_factor * max(L, W, H) * 0.7
                ),
            ),
        ),
        height=800,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    return fig

def _nice_step(n: int, target_labels: int = 7) -> int:
    """
    choose a integer step (1/2/5×10^k) to label sliders

    args:
        n: number of items
        target_labels: rough number of labels desired
    returns:
        integer step size
    """
    raw = max(1, int(math.ceil(n / max(target_labels, 2))))
    mag = 10 ** int(math.floor(math.log10(raw)))
    for m in (1, 2, 5, 10):
        s = m * mag
        if s >= raw:
            return s
    return raw

def make_slider_marks_from_vmax(vmax: int) -> dict:
    """
    build sparse slider marks from a maximum index

    args:
        vmax: maximum step index
    returns:
        dict of {index: label} for dash slider marks
    """
    vmax = int(vmax or 0)
    n = vmax + 1
    if n <= 15:
        return {i: str(i) for i in range(n)}
    step = _nice_step(n, target_labels=7)
    marks = {0: "0"}
    for i in range(step, vmax, step):
        marks[i] = str(i)
    marks[vmax] = str(vmax)
    return marks

#===========================================================
#metrics card rendering
#===========================================================

def render_metrics_card(_metrics: dict, df: pd.DataFrame, mode: str) -> html.Div:
    """
    render a compact mission metrics card consistent with validator logic

    args:
        _metrics: metrics dict (may be ignored; computed here for robustness)
        df: dataframe from wps_to_df
        mode: "2D" or "3D" to select validator behavior
    returns:
        dash html.Div with summary grids and status badges
    """
    if df is None or df.empty:
        return html.Div(
            [
                html.Strong("Mission summary"),
                html.Div("Click “Generate Path” to see metrics.", style={"color": "#666", "marginTop": "6px"}),
            ]
        )

    # ---------- helpers ----------
    def _fmt_sec(s: float) -> str:
        try:
            s = float(s)
        except Exception:
            return str(s)
        return f"{s:.1f} s" if s < 120 else f"{s/60:.1f} min"

    def _badge(ok: bool, good="OK", bad="Fail"):
        color = "#10b981" if ok else "#ef4444"
        bg = "#ecfdf5" if ok else "#fef2f2"
        txt = good if ok else bad
        return html.Span(
            txt,
            style={
                "display": "inline-block",
                "padding": "2px 8px",
                "borderRadius": "999px",
                "fontSize": "12px",
                "fontWeight": 600,
                "color": color,
                "background": bg,
            },
        )

    def _grid(items):
        #simple two-column key/value grid
        return html.Div(
            [
                html.Div(
                    [html.Span(f"{k}:"), html.Span(v, style={"fontWeight": 600})],
                    style={"display": "flex", "justifyContent": "space-between", "gap": "8px"},
                )
                for k, v in items
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "columnGap": "18px", "rowGap": "6px"},
        )

    def is_right_angle(v1, v2, tol_deg=15.0):
        #detect 90° between consecutive horizontal legs
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-12 or n2 < 1e-12:
            return False
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.degrees(math.acos(cosang))
        return (90.0 - tol_deg) <= ang <= (90.0 + tol_deg)

    # ---------- rebuild waypoints from df ----------
    wps: List[Waypoint] = [
        Waypoint(
            float(r["x"]),
            float(r["y"]),
            float(r["z"]),
            float(r.get("gimbal", 0.0)),
            float(r.get("speed", DEFAULT_DRONE.speed)),
            float(r.get("hold", 0.0)),
        )
        for _, r in df.iterrows()
    ]
    n_pts = len(wps)

    # ---------- constants ----------
    drone = DEFAULT_DRONE
    include_takeoff_landing = True
    ground_z = 0.0
    controller_wp_pause_s = 2.0       
    fixed_pre_post_pauses_s = 12.0     
    v_horiz = drone.speed
    v_vert = getattr(drone, "vertical_speed", None) or drone.speed
    base_power = 60.0
    base_yaw_deg_s = 120.0
    power_rot = 46.67
    yaw_rate_large = base_yaw_deg_s * (power_rot / base_power)
    t90 = 90.0 / max(yaw_rate_large, 1e-6)
    eps_z = 1e-9

    # ---------- decide 2D vs 3D by Z changes ----------
    z_vals = [wp.z for wp in wps]
    z_runs = 1
    for a, b in zip(z_vals, z_vals[1:]):
        if abs(a - b) > eps_z:
            z_runs += 1
    use_3d = (mode == "3D") or (z_runs > 1)

    # ---------- compute metrics ----------
    if not use_3d:
        # --- 2D validator ---
        total_distance = 0.0
        travel_time = 0.0

        distance_takeoff = 0.0
        if include_takeoff_landing and wps[0].z != ground_z:
            distance_takeoff = abs(wps[0].z - ground_z)
            travel_time += distance_takeoff / v_vert
            total_distance += distance_takeoff

        distance_cruise = 0.0
        leg_vecs = []
        for wp1, wp2 in zip(wps, wps[1:]):
            dx, dy = wp2.x - wp1.x, wp2.y - wp1.y
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                continue
            d2 = math.hypot(dx, dy)
            distance_cruise += d2
            total_distance += d2
            travel_time += d2 / max(v_horiz, 1e-9)
            leg_vecs.append((dx, dy))

        num_corners = sum(1 for v1, v2 in zip(leg_vecs, leg_vecs[1:]) if is_right_angle(v1, v2))
        turn_time = num_corners * t90
        travel_time += turn_time

        distance_landing = 0.0
        if include_takeoff_landing and wps[-1].z != ground_z:
            distance_landing = abs(wps[-1].z - ground_z)
            travel_time += distance_landing / v_vert
            total_distance += distance_landing

        hover_time = sum(getattr(wp, "hold_time", 0.0) for wp in wps)
        controller_pause_time = controller_wp_pause_s * n_pts
        fixed_overhead_time = fixed_pre_post_pauses_s

        total_time = travel_time + hover_time + controller_pause_time + fixed_overhead_time
        battery_used_pct = (total_time / drone.max_battery_time) * 100.0

        battery_ok = battery_used_pct <= 100.0
        battery_warning = battery_used_pct >= (drone.battery_warning_threshold * 100.0)
        distance_ok = total_distance <= drone.max_distance

        metrics = dict(
            distance=total_distance,
            distance_takeoff=distance_takeoff,
            distance_cruise=distance_cruise,
            distance_landing=distance_landing,
            num_corners=num_corners,
            t90_used_s=t90,
            turn_time_90s=turn_time,
            travel_time=travel_time,
            hover_time=hover_time,
            controller_pause_time=controller_pause_time,
            fixed_overhead_time=fixed_overhead_time,
            total_time=total_time,
            battery_usage_percent=battery_used_pct,
            battery_ok=battery_ok,
            battery_warning=battery_warning,
            distance_ok=distance_ok,
        )
    else:
        # --- 3D validator ---
        # build layers (contiguous runs of equal Z)
        layers: List[Tuple[float, int, int]] = []
        start_idx = 0
        current_z = wps[0].z
        for i in range(1, n_pts):
            if abs(wps[i].z - current_z) > eps_z:
                layers.append((current_z, start_idx, i - 1))
                start_idx = i
                current_z = wps[i].z
        layers.append((current_z, start_idx, n_pts - 1))

        layer_stats = [
            {
                "z": z,
                "idx_start": s,
                "idx_end": e,
                "num_waypoints": (e - s + 1),
                "distance_2d": 0.0,
                "distance_3d": 0.0,
                "travel_time": 0.0,
                "hover_time": sum(getattr(wp, "hold_time", 0.0) for wp in wps[s : e + 1]),
                "num_corners": 0,
                "turn_time_90s": 0.0,
                "total_time": 0.0,
                "battery_usage_percent": 0.0,
            }
            for (z, s, e) in layers
        ]

        def which_layer(idx: int) -> int:
            for li, (_, s, e) in enumerate(layers):
                if s <= idx <= e:
                    return li
            return -1

        total_distance = 0.0
        distance_takeoff = 0.0
        distance_cruise = 0.0
        distance_landing = 0.0
        travel_time = 0.0
        hover_time = sum(getattr(wp, "hold_time", 0.0) for wp in wps)

        if include_takeoff_landing and abs(wps[0].z - ground_z) > eps_z:
            distance_takeoff = abs(wps[0].z - ground_z)
            travel_time += distance_takeoff / v_vert
            total_distance += distance_takeoff

        vertical_seams_count = 0
        vertical_seams_distance = 0.0
        vertical_seams_time = 0.0

        layer_leg_vecs: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(layers))}

        for i, (wp1, wp2) in enumerate(zip(wps, wps[1:])):
            dx, dy, dz = wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z
            d2 = math.hypot(dx, dy)
            d3 = math.sqrt(dx * dx + dy * dy + dz * dz)

            li1 = which_layer(i)
            li2 = which_layer(i + 1)

            total_distance += d3

            if li1 == li2 and li1 != -1 and abs(dz) <= eps_z:
                distance_cruise += d2
                layer_stats[li1]["distance_2d"] += d2
                layer_stats[li1]["distance_3d"] += d3
                t_seg = d2 / max(v_horiz, 1e-9)
                layer_stats[li1]["travel_time"] += t_seg
                travel_time += t_seg
                if d2 > 1e-12:
                    layer_leg_vecs[li1].append((dx, dy))
            else:
                vertical_seams_count += 1
                vertical_seams_distance += d3
                distance_cruise += d3
                t_seam = 0.0
                if d2 > 0.0:
                    t_seam += d2 / max(v_horiz, 1e-9)
                if abs(dz) > 0.0:
                    t_seam += abs(dz) / max(v_vert, 1e-9)
                vertical_seams_time += t_seam
                travel_time += t_seam

        if include_takeoff_landing and abs(wps[-1].z - ground_z) > eps_z:
            distance_landing = abs(wps[-1].z - ground_z)
            travel_time += distance_landing / v_vert
            total_distance += distance_landing

        num_corners_total = 0
        turn_time_total = 0.0
        for li, vecs in layer_leg_vecs.items():
            n_corners = sum(1 for v1, v2 in zip(vecs, vecs[1:]) if is_right_angle(v1, v2))
            t_layer_turn = n_corners * t90
            layer_stats[li]["num_corners"] = n_corners
            layer_stats[li]["turn_time_90s"] = t_layer_turn
            num_corners_total += n_corners
            turn_time_total += t_layer_turn
        travel_time += turn_time_total

        controller_pause_time = controller_wp_pause_s * n_pts
        fixed_overhead_time = fixed_pre_post_pauses_s

        total_time = travel_time + hover_time + controller_pause_time + fixed_overhead_time
        battery_used_pct = (total_time / drone.max_battery_time) * 100.0

        for ls in layer_stats:
            ls["total_time"] = ls["travel_time"] + ls["hover_time"]
            ls["battery_usage_percent"] = (ls["total_time"] / drone.max_battery_time) * 100.0

        battery_ok = battery_used_pct <= 100.0
        battery_warning = battery_used_pct >= (drone.battery_warning_threshold * 100.0)
        distance_ok = total_distance <= drone.max_distance

        # ---- CALIBRATION (3D ONLY) ----
        CALIB_EXTRA_FIXED_S  = 0   
        CALIB_EXTRA_PER_WP_S = 5   
        CALIB_TARGET_TOTAL_S = None  

        if CALIB_TARGET_TOTAL_S is not None:
            _extra = max(0.0, float(CALIB_TARGET_TOTAL_S) - float(total_time))
        else:
            _extra = float(CALIB_EXTRA_FIXED_S) + float(CALIB_EXTRA_PER_WP_S) * n_pts

        if _extra > 0.0:
            fixed_overhead_time += _extra
            total_time          += _extra
            battery_used_pct     = (total_time / drone.max_battery_time) * 100.0
            battery_ok           = battery_used_pct <= 100.0
            battery_warning      = battery_used_pct >= (drone.battery_warning_threshold * 100.0)

        metrics = dict(
            distance=total_distance,
            distance_takeoff=distance_takeoff,
            distance_cruise=distance_cruise,
            distance_landing=distance_landing,
            travel_time=travel_time,
            hover_time=hover_time,
            controller_pause_time=controller_pause_time,
            fixed_overhead_time=fixed_overhead_time,
            total_time=total_time,
            battery_usage_percent=battery_used_pct,
            battery_ok=battery_ok,
            battery_warning=battery_warning,
            distance_ok=distance_ok,
            vertical_seams_count=vertical_seams_count,
            vertical_seams_distance=vertical_seams_distance,
            vertical_seams_time=vertical_seams_time,
            num_corners=num_corners_total,
            t90_used_s=t90,
            turn_time_90s=turn_time_total,
            layers=layer_stats,
        )

    # ---------- render card ----------
    n_wps_display = len(df)
    d_take = metrics["distance_takeoff"]
    d_cru = metrics["distance_cruise"]
    d_land = metrics["distance_landing"]

    primary_facts = [
        ("Waypoints", f"{n_wps_display}"),
        ("Total distance", f"{metrics['distance']:.1f} m"),
        ("Total time", _fmt_sec(metrics["total_time"])),
        ("Battery usage", f"{metrics['battery_usage_percent']:.0f}%"),
    ]
    secondary_facts = [
        ("Travel time", _fmt_sec(metrics["travel_time"])),
        ("Hover time", _fmt_sec(metrics["hover_time"])),
        ("Ctrl. pauses", _fmt_sec(metrics["controller_pause_time"])),
        ("Fixed overhead", _fmt_sec(metrics["fixed_overhead_time"])),
        ("Take-off dist", f"{d_take:.1f} m"),
        ("Cruise dist", f"{d_cru:.1f} m"),
        ("Landing dist", f"{d_land:.1f} m"),
        ("90° turns", f"{int(metrics.get('num_corners', 0))}"),
        ("t90 per turn", f"{metrics.get('t90_used_s', 0.0):.2f} s"),
        ("Turn time total", _fmt_sec(metrics.get("turn_time_90s", 0.0))),
    ]
    if "vertical_seams_count" in metrics:
        secondary_facts.extend(
            [
                ("Vertical seams", f"{int(metrics['vertical_seams_count'])}"),
                ("Seams distance", f"{metrics['vertical_seams_distance']:.1f} m"),
                ("Seams time", _fmt_sec(metrics["vertical_seams_time"])),
            ]
        )

    breakdown_line = html.Div(
        [
            html.Span("Breakdown:"),
            html.Span(
                f"Take-off {d_take:.1f} m • Cruise {d_cru:.1f} m • Landing {d_land:.1f} m",
                style={"fontWeight": 600},
            ),
        ],
        style={
            "gridColumn": "1 / -1",
            "borderTop": "1px solid #eaeaea",
            "paddingTop": "6px",
            "marginTop": "6px",
            "display": "flex",
            "justifyContent": "space-between",
        },
    )
    status_line = html.Div(
        [
            html.Span("Battery: "),
            _badge(bool(metrics["battery_ok"]), "OK", "Insufficient"),
            html.Span("Warning: "),
            _badge(not bool(metrics["battery_warning"]), "No", "Yes"),
            html.Span("Distance: "),
            _badge(bool(metrics["distance_ok"]), "OK", "Limit Exceeded"),
        ],
        style={"gridColumn": "1 / -1", "display": "flex", "gap": "8px", "alignItems": "center"},
    )

    return html.Div(
        [html.Strong("Mission summary"), _grid(primary_facts), _grid(secondary_facts), breakdown_line, status_line],
        style={"display": "grid", "rowGap": "8px"},
    )


def _decode_upload(contents: str) -> bytes:
    """
    decode a dash upload content string (data url or bare base64) into bytes

    args:
        contents: string returned by dcc.Upload or raw base64)
    returns:
        decoded bytes or None if input is falsy
    """
    if not contents:
        return None
    try:
        header, b64data = contents.split(",", 1)
    except ValueError:
        b64data = contents
    return base64.b64decode(b64data)


def _make_json_safe(o):
    """
    recursively convert to JSON-safe types.
    """
    if isinstance(o, (bytes, bytearray)):
        return base64.b64encode(o).decode("ascii")
    if is_dataclass(o):
        return _make_json_safe(asdict(o))
    if isinstance(o, np.generic):                 
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_list()
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): _make_json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_make_json_safe(v) for v in o]
    return o


def _ensure_text(v):
    """
    ensure a string representation; try utf-8 decode for bytes, else base64
    """
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8")
        except Exception:
            return base64.b64encode(v).decode("ascii")
    return v if isinstance(v, str) else str(v)


def _ensure_b64(v):
    """
    ensure a base64 string for storage/transport
    """
    if isinstance(v, str):
        return v
    if isinstance(v, (bytes, bytearray)):
        return base64.b64encode(v).decode("ascii")
    # last-resort fallback
    return _ensure_text(v)


# ==============================================================
# callbacks 
# ==============================================================

@app.callback(
    Output(IDS["store_report_bundle"], "data"),
    Output(IDS["report_output"], "children"),
    Input(IDS["btn_run_report"], "n_clicks"),
    State(IDS["upload_log"], "contents"),
    State(IDS["upload_log"], "filename"),
    State(IDS["upload_path"], "contents"),
    State(IDS["upload_path"], "filename"),
    State(IDS["store_cfg"], "data"),
    prevent_initial_call=True,
)
def run_report(n_clicks, log_contents, log_name, path_contents, path_name, cfg):
    if not n_clicks:
        return no_update, no_update
    if not log_contents:
        return no_update, html.Div("Please upload a log file.", style={"color": "crimson"})

    #---------- decode uploads ----------
    try:
        log_bytes = _decode_upload(log_contents)
        flight_bytes = _decode_upload(path_contents) if path_contents else None
    except Exception as e:
        return no_update, html.Div(f"Failed to read uploads: {e}", style={"color": "crimson"})

    #---------- optional workspace dims ----------
    L = (cfg or {}).get("L"); W = (cfg or {}).get("W"); H = (cfg or {}).get("H")
    dims = {"x": (0.0, L), "y": (0.0, W), "z": (0.0, H)} if all(v is not None for v in (L, W, H)) else None

    #---------- run the generator ----------
    try:
        bundle = LogReportGenerator.generate_report_bundle(
            log_bytes=log_bytes,
            flightpath_bytes=flight_bytes,
            workspace_dims=dims
        )
    except Exception as e:
        return no_update, html.Div(f"Report failed: {e}", style={"color": "crimson"})

    report_md = _ensure_text(bundle.get("report_md") or "")
    #figures as base64 strings
    figs_in = bundle.get("figures") or {}
    figs = {name: _ensure_b64(img) for name, img in figs_in.items()}
    #zip_bytes -> zip_b64
    if isinstance(bundle.get("zip_bytes"), (bytes, bytearray)):
        bundle["zip_b64"] = base64.b64encode(bundle["zip_bytes"]).decode("ascii")
        bundle.pop("zip_bytes", None)

    bundle["report_md"] = report_md
    bundle["figures"] = figs

    #---------- JSON-safety ----------
    #bytes -> b64 strings, numpy -> native, pandas -> dicts, datetimes -> iso
    safe_bundle = _make_json_safe(bundle)

    #debug
    def _find_bytes(o, path="$"):
        if isinstance(o, (bytes, bytearray)):
            return [path]
        if isinstance(o, dict):
            out = []
            for k, v in o.items():
                out.extend(_find_bytes(v, f"{path}.{k}"))
            return out
        if isinstance(o, (list, tuple, set)):
            out = []
            for i, v in enumerate(o):
                out.extend(_find_bytes(v, f"{path}[{i}]"))
            return out
        return []

    offenders = _find_bytes(safe_bundle)
    if offenders:
        return no_update, html.Div(
            f"Internal error: raw bytes found at {offenders[:5]}",
            style={"color": "crimson"}
        )

    #store as a JSON string to avoid Plotly’s encoder problems
    safe_bundle_json = json.dumps(safe_bundle)

    #---------- building the UI ----------
    sections = safe_bundle.get("sections") or []
    ordered = []
    for s in sections:
        for fname in s.get("figs", []):
            if fname in figs and fname not in ordered:
                ordered.append(fname)
    for fname in figs:
        if fname not in ordered:
            ordered.append(fname)

    fig_blocks = []
    for fname in ordered:
        b64 = figs.get(fname)
        if not b64:
            continue
        fig_blocks.append(
            html.Details(
                open=False,
                children=[
                    html.Summary(fname),
                    html.Img(
                        src="data:image/png;base64," + b64,
                        style={
                            "maxWidth": "100%",
                            "borderRadius": "10px",
                            "marginTop": "8px",
                            "boxShadow": "0 1px 6px rgba(0,0,0,0.08)",
                        },
                    ),
                ],
                style={"margin": "10px 0"},
            )
        )

    output_children = [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(f"Log: {log_name or '—'}", style={"fontSize": 12, "color": "#555"}),
                                html.Div(f"Flightpath: {path_name or '—'}", style={"fontSize": 12, "color": "#555"}),
                            ],
                            style={"marginBottom": "8px"},
                        ),
                        dcc.Markdown(report_md, link_target="_blank"),
                    ],
                    style={"background": "white", "border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "16px"},
                ),
                html.Div(fig_blocks, style={"marginTop": "12px"}),
            ],
            style={"display": "grid", "rowGap": "12px"},
        )
    ]

    return safe_bundle_json, output_children

@app.callback(
    Output("3d-only", "style"),
    Input("scan-mode", "value"),
)
def toggle_3d(section_mode):
    return {"display": "block", "marginTop": "6px"} if section_mode == "3D" else {"display": "none"}

@app.callback(
    Output(IDS["store_wps"], "data"),
    Output(IDS["store_metrics"], "data"),
    Output(IDS["store_cfg"], "data"),
    Output(IDS["slider"], "max"),
    Output(IDS["slider"], "marks"),
    Output(IDS["graph"], "figure"),
    Output("metrics-card", "children"),
    Output(IDS["error"], "children"),
    Input(IDS["btn_generate"], "n_clicks"),
    Input(IDS["btn_reset"], "n_clicks"),          
    State("length", "value"),
    State("width", "value"),
    State("height", "value"),
    State("scan-mode", "value"),
    State("overlap", "value"),
    State("wall-offset", "value"),
    State("clearance", "value"),
    State("altitude", "value"),
    State("turns", "value"),
    State("layers", "value"),
    State("top-cap", "value"),
    State({"type": "dz", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def generate_or_reset(nc_generate, nc_reset, L, W, H, mode, overlap, wall_off, clearance, alt, turns, layers, z_cap, dz_list):
    """
    generate waypoints and figures, or reset ui if reset button triggered
    """
    #----- handle reset with no duplicates -----
    trig = ctx.triggered_id
    if trig == IDS["btn_reset"]:
        empty_df = pd.DataFrame(columns=["x", "y", "z", "step"])
        empty_fig = make_2d_fig(empty_df, DEFAULT_ROOM["length"], DEFAULT_ROOM["width"], 0)
        card = html.Div(
            [html.Strong("Mission summary"),
             html.Div("Click “Generate Path” to see metrics.", style={"color": "#666", "marginTop": "6px"})]
        )
        return None, None, {"scan_mode": "2D"}, 0, {}, empty_fig, card, ""

    #----- generate logic -----
    try:
        #input validation
        for name, val in [("Length", L), ("Width", W), ("Height", H), ("Altitude", alt)]:
            if val is None or val <= 0:
                return no_update, no_update, no_update, no_update, no_update, no_update, no_update, f"Invalid {name}"
        if not (0.0 <= overlap < 1.0):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, "Overlap must be in [0,1)."

        dims = (float(L), float(W), float(H))
        drone = DEFAULT_DRONE

        if mode == "2D":
            base = generate_spiral_scan(
                drone, dims, float(alt), float(overlap),
                float(wall_off or 0), float(clearance or 0), int(turns or 3)
            )
            wps = reduce_waypoints_collinear(base)
            _feasible, metrics = validate_mission(drone, wps)
            df = wps_to_df(wps)
            fig = make_2d_fig(df, L, W, 0 if df.empty else int(df["step"].max()))
        else:
            base = generate_spiral_scan(
                drone, dims, float(alt), float(overlap),
                float(wall_off or 0), float(clearance or 0), int(turns or 3)
            )

            n_layers = int(layers or 1)
            if n_layers < 1:
                n_layers = 1

            #build absolute heights from inputs, fallback to 'alt' if missing/invalid
            heights = []
            fallback_h = float(alt)
            for i in range(n_layers):
                h = fallback_h
                if isinstance(dz_list, list) and i < len(dz_list) and dz_list[i] is not None:
                    try:
                        hv = float(dz_list[i])
                        if hv >= 0.0:
                            h = hv
                    except Exception:
                        pass
                h = max(drone.min_altitude, h)
                heights.append(h)

            #optional top cap
            if z_cap is not None:
                try:
                    zcap = float(z_cap)
                    heights = [z for z in heights if z <= zcap]
                except Exception:
                    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, "Invalid Z limit."

            if not heights:
                return no_update, no_update, no_update, no_update, no_update, no_update, no_update, "No valid layers under z cap."

            wps3d = stacked_spiral(tuple(heights), base)
            wps = reduce_waypoints_collinear(wps3d)
            _feasible, metrics = validate_mission_with_layers(drone, wps)
            df = wps_to_df(wps)
            fig = make_3d_fig(df, L, W, H, 0 if df.empty else int(df["step"].max()))

        max_step = 0 if df.empty else int(df["step"].max())
        marks = make_slider_marks_from_vmax(max_step)
        cfg = {"scan_mode": mode, "L": L, "W": W, "H": H}
        card_children = render_metrics_card(metrics, df, mode)
        return df.to_dict("records"), metrics, cfg, max_step, marks, fig, card_children, ""

    except Exception as e:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, f"Error: {e}"



@app.callback(
    Output(IDS["graph"], "figure", allow_duplicate=True),
    Input(IDS["slider"], "value"),
    State(IDS["store_wps"], "data"),
    State(IDS["store_cfg"], "data"),
    prevent_initial_call=True,
)
def scrub(step_value, wps_data, cfg):
    """
    update plot to show drone position up to a given step
    allows the user to scrub through the mission timeline
    """
    if not wps_data or not cfg:
        return go.Figure()
    df = pd.DataFrame(wps_data)
    mode = cfg.get("scan_mode", "2D")
    L, W, H = cfg.get("L", 0.0), cfg.get("W", 0.0), cfg.get("H", 0.0)
    step_idx = int(step_value or 0)
    if mode == "2D":
        fig = make_2d_fig(df, L, W, step_idx)
    else:
        fig = make_3d_fig(df, L, W, H, step_idx)
    return fig


@app.callback(
    Output("dz-container", "children"),
    Input("layers", "value"),
    State("altitude", "value"),
    prevent_initial_call=False,
)
def render_dz_inputs(n_layers, alt_val):
    """
    dynamically render altitude input boxes for each layer
    """
    try:
        n = int(n_layers or 1)
    except Exception:
        n = 1
    if n <= 0:
        n = 1

    default_val = float(alt_val or 1.0)

    return [
        html.Label("Layer heights (m) — one value per layer", style={"fontSize": 12, "color": "#555"}),
        html.Div(
            [
                dcc.Input(
                    id={"type": "dz", "index": i},
                    type="number",
                    value=default_val,
                    min=0.0,
                    step=0.05,
                    placeholder=f"z layer {i+1}",
                    style={"width": "100%"},
                )
                for i in range(n)
            ],
            style={"display": "grid", "gridTemplateColumns": "repeat(6,1fr)", "gap": "8px"},
        ),
    ]




@app.callback(
    Output(IDS["tick"], "disabled"),
    Input(IDS["play"], "n_clicks"),
    Input(IDS["pause"], "n_clicks"),
    prevent_initial_call=True,
)
def toggle_timer(n_play, n_pause):
    """
    toggle the timeline animation play/pause button
    """
    trig = ctx.triggered_id
    if trig == IDS["play"]:
        return False
    if trig == IDS["pause"]:
        return True
    return True


@app.callback(
    Output(IDS["slider"], "value"),
    Input(IDS["tick"], "n_intervals"),
    State(IDS["slider"], "value"),
    State(IDS["slider"], "max"),
)
def advance(_n, val, vmax):
    """
    advance slider automatically during play mode
    """
    if vmax is None:
        return no_update
    val = int(val or 0)
    vmax = int(vmax or 0)
    return 0 if val >= vmax else val + 1


#--- downloads  ----

@app.callback(
    Output(IDS["dl_csv"], "data"),
    Input("btn-csv", "n_clicks"),
    State(IDS["store_wps"], "data"),
    prevent_initial_call=True,
)
def download_csv(nc, wps_data):
    if not wps_data:
        return no_update
    df = pd.DataFrame(wps_data)
    wps = [Waypoint(row["x"], row["y"], row["z"], row.get("gimbal", 0.0), row.get("speed", 0.0), row.get("hold", 0.0)) for _, row in df.iterrows()]
    csv_text = render_marvelmind_csv(wps)
    return dict(content=csv_text, filename="waypoints.csv")


@app.callback(
    Output(IDS["dl_txt"], "data"),
    Input("btn-txt", "n_clicks"),
    State(IDS["store_wps"], "data"),
    prevent_initial_call=True,
)
def download_txt(nc, wps_data):
    if not wps_data:
        return no_update
    df = pd.DataFrame(wps_data)
    wps = [Waypoint(row["x"], row["y"], row["z"], row.get("gimbal", 0.0), row.get("speed", 0.0), row.get("hold", 0.0)) for _, row in df.iterrows()]
    txt = render_txt(wps)
    return dict(content=txt, filename="waypoints.txt")


@app.callback(
    Output(IDS["dl_dpt"], "data"),
    Input("btn-dpt", "n_clicks"),
    State(IDS["store_wps"], "data"),
    prevent_initial_call=True,
)
def download_dpt(nc, wps_data):
    if not wps_data:
        return no_update
    df = pd.DataFrame(wps_data)
    wps = [Waypoint(row["x"], row["y"], row["z"], row.get("gimbal", 0.0), row.get("speed", 0.0), row.get("hold", 0.0)) for _, row in df.iterrows()]
    dpt_text = render_dpt(wps, DEFAULT_DRONE)
    return dict(content=dpt_text, filename="mission.dpt")

@app.callback(
    Output(IDS["btn_run_report"], "disabled"),
    Output(IDS["btn_run_report"], "style"),
    Input(IDS["upload_log"], "contents"),
    Input(IDS["upload_path"], "contents"),
)
def toggle_run_button(log_contents, path_contents):
    """
    enable or disable report generation button based on uploaded files
    """
    base_style = {
        "padding": "12px 18px",
        "background": "#10b981",
        "color": "white",
        "border": "none",
        "borderRadius": "12px",
        "fontWeight": 700,
        "fontSize": "16px",
        "boxShadow": "0 2px 6px rgba(16,185,129,0.3)",
        "cursor": "pointer",
        "opacity": 1.0,
    }
    ready = bool(log_contents) and bool(path_contents)
    if not ready:
        base_style["cursor"] = "not-allowed"
        base_style["opacity"] = 0.5
    return (not ready), base_style


@app.callback(
    Output(IDS["upload_log"], "children"),
    Output(IDS["upload_path"], "children"),
    Input(IDS["upload_log"], "filename"),
    Input(IDS["upload_path"], "filename"),
)
def _show_uploaded_names(log_name, path_name):
    """
    show the uploaded file names or placeholder text
    """
    def box(title, fname):
        return html.Div([
            html.Strong(title),
            html.Div(
                (fname or "Drag & drop or click to upload"),
                style={"color": "#666", "fontSize": "12px"}
            ),
        ])
    return box("Log file (.csv)", log_name), box("Flightpath (.csv)", path_name)

@app.callback(
    Output(IDS["btn_report_link"], "style"),
    Input(IDS["store_wps"], "data"),
)
def show_report_link(wps):
    return {"display": "inline-block"} if wps else {"display": "none"}

@app.callback(
    Output("home-page", "style"),
    Output("report-page", "style"),
    Input("url", "pathname"),
)
def route(pathname):
    """
    toggle between home page and report page based on url path
    """
    if pathname == "/report":
        return HIDDEN, REPORT_STYLE
    # default = home
    return HOME_STYLE, HIDDEN

# ============================================================
# run locally 
# ============================================================
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False, host="127.0.0.1", port=8051)

