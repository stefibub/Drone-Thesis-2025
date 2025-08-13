import io
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import base64, pathlib
from dash.dependencies import ALL

# ============================================================
# ========  YOUR ORIGINAL DATA STRUCTURES / LOGIC  ==========
#  (kept as-is as much as possible; tiny wrappers are below)
# ============================================================

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
    inset_x = wall_offset + clearance
    inset_y = clearance
    hold = 1.0 / drone.fps + drone.hover_buffer

    min_x = inset_x
    max_x = dims[0] - inset_x
    min_y = inset_y
    max_y = dims[1] - inset_y

    if min_x >= max_x or min_y >= max_y:
        return []

    altitude = max(float(altitude), float(clearance or 0.0), float(drone.min_altitude))

    fp_long, fp_short = calculate_footprint(drone, altitude)
    avg_fp = (fp_long + fp_short) / 2.0
    nominal_step = avg_fp * (1 - overlap)
    speed = calculate_speed(avg_fp, overlap, drone.fps)

    span_x = max_x - min_x
    span_y = max_y - min_y
    min_span = min(span_x, span_y)
    step = min(nominal_step, min_span / (2 * target_turns + 1))

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    x, y = cx, cy

    wps: List[Waypoint] = [Waypoint(x, y, altitude, 0.0, speed, hold)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    layer = 1
    dir_idx = 0

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
        cross = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )
        cross_norm = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        if cross_norm < eps and dot > 0:
            continue
        reduced.append(curr)
    reduced.append(waypoints[-1])
    return reduced


def stacked_spiral(heights: Tuple[float], base_waypoints: List[Waypoint]) -> List[Waypoint]:
    if not heights or not base_waypoints:
        return []

    def clone_with_z(seq: List[Waypoint], z: float) -> List[Waypoint]:
        return [Waypoint(w.x, w.y, z, w.gimbal_pitch, w.speed, w.hold_time) for w in seq]

    out: List[Waypoint] = []
    forward = base_waypoints
    reverse = list(reversed(base_waypoints))

    current_layer = clone_with_z(forward, heights[0])
    out.extend(current_layer)

    base_start_xy = (forward[0].x, forward[0].y)
    base_end_xy = (forward[-1].x, forward[-1].y)

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


def validate_mission(
    drone: DroneConfig,
    waypoints: List[Waypoint],
    include_takeoff_landing: bool = True,
    ground_z: float = 0.0,
    controller_wp_pause_s: float = 2.0,
    fixed_pre_post_pauses_s: float = 12.0,
) -> Tuple[bool, dict]:
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

    base_power = 60.0
    base_yaw_deg_s = 120.0
    power_rot = 46.67
    yaw_rate_large = base_yaw_deg_s * (power_rot / base_power)
    t90 = 90.0 / max(yaw_rate_large, 1e-6)

    total_distance = 0.0
    travel_time = 0.0

    distance_takeoff = 0.0
    if include_takeoff_landing and waypoints[0].z != ground_z:
        distance_takeoff = abs(waypoints[0].z - ground_z)
        travel_time += distance_takeoff / v_vert
        total_distance += distance_takeoff

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
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return False
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.degrees(math.acos(cosang))
        return (90.0 - tol_deg) <= ang <= (90.0 + tol_deg)

    num_corners = sum(1 for v1, v2 in zip(leg_vecs, leg_vecs[1:]) if is_right_angle(v1, v2))
    turn_time = num_corners * t90
    travel_time += turn_time

    distance_landing = 0.0
    if include_takeoff_landing and waypoints[-1].z != ground_z:
        distance_landing = abs(waypoints[-1].z - ground_z)
        travel_time += distance_landing / v_vert
        total_distance += distance_landing

    hover_time = sum(getattr(wp, "hold_time", 0.0) for wp in waypoints)
    controller_pause_time = controller_wp_pause_s * len(waypoints)
    fixed_overhead_time = fixed_pre_post_pauses_s

    total_time = travel_time + hover_time + controller_pause_time + fixed_overhead_time
    battery_used_pct = (total_time / drone.max_battery_time) * 100.0

    battery_ok = battery_used_pct <= 100.0
    battery_warning = battery_used_pct >= (drone.battery_warning_threshold * 100.0)
    distance_ok = total_distance <= drone.max_distance
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


def validate_mission_with_layers(
    drone: DroneConfig,
    waypoints: List[Waypoint],
    include_takeoff_landing: bool = True,
    ground_z: float = 0.0,
    controller_wp_pause_s: float = 2.0,
    fixed_pre_post_pauses_s: float = 12.0,
) -> Tuple[bool, dict]:
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

    base_power = 60.0
    base_yaw_deg_s = 120.0
    power_rot = 46.67
    yaw_rate_large = base_yaw_deg_s * (power_rot / base_power)
    t90 = 90.0 / max(yaw_rate_large, 1e-6)

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
        for li, (_, s, e) in enumerate(layers):
            if s <= idx <= e:
                return li
        return -1

    def is_right_angle(v1, v2, tol_deg=15.0) -> bool:
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

    for ls in layer_stats:
        ls["total_time"] = ls["travel_time"] + ls["hover_time"]
        ls["battery_usage_percent"] = (ls["total_time"] / drone.max_battery_time) * 100.0

    battery_ok = battery_used_pct <= 100.0
    battery_warning = battery_used_pct >= (drone.battery_warning_threshold * 100.0)
    distance_ok = total_distance <= drone.max_distance
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


# -------- Original file-based exporters (KEPT, but unused in hosted mode) --------
# def export_to_marvelmind(waypoints: List[Waypoint], filename: str): ...
# def export_to_dpt(waypoints: List[Waypoint], drone_config: DroneConfig, filename: str): ...

# ===== Hosted-friendly in-memory export (no disk writes) =====

def render_marvelmind_csv(waypoints: List[Waypoint]) -> str:
    # index, x, y, z, HoldTime, GimblePitch
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


# ============================================================
# ===============  DASH APP (HOSTED-READY)  ==================
# ============================================================

app = Dash(__name__, suppress_callback_exceptions=True, title="Room Scan Flightpath")
server = app.server  # for Gunicorn / PaaS
logo_path = pathlib.Path("/Users/stefaniaconte/Desktop/Drone-Thesis-2025/ucd_logo.jpg")
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()

# ---- Component IDs ----
IDS = dict(
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
)

# ---- Defaults ----
DEFAULT_DRONE = DroneConfig(
    weight=0.292,
    max_battery_time=1380.0,
    max_distance=1e6,  # effectively unlimited unless you want a hard cap
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
INITIAL_FIG = go.Figure()
INITIAL_FIG.add_shape(type="rect",
                      x0=0, y0=0,
                      x1=DEFAULT_ROOM["length"], y1=DEFAULT_ROOM["width"],
                      line=dict(dash="dot"))
INITIAL_FIG.update_xaxes(range=[0, DEFAULT_ROOM["length"]], autorange=False, fixedrange=True, title_text="x (m)")
INITIAL_FIG.update_yaxes(range=[0, DEFAULT_ROOM["width"]],  autorange=False, fixedrange=True,
                         scaleanchor="x", scaleratio=1, title_text="y (m)")
INITIAL_FIG.update_layout(height=600, margin=dict(l=40, r=20, t=20, b=40))
TITLE_STYLE = {
    "margin": "0 0 8px 0",   # remove default top margin so baselines align
    "fontSize": "20px",
    "fontWeight": "600",
    "letterSpacing": "0.2px",
}
TIP_ICON_STYLE = {"cursor": "help", "color": "#888", "marginLeft": "6px", "fontWeight": 600}

# ---- Layout ----
app.layout = html.Div(
    className="app",
    children=[
        dcc.Store(id=IDS["store_wps"]),
        dcc.Store(id=IDS["store_metrics"]),
        dcc.Store(id=IDS["store_cfg"], data={"scan_mode": "2D"}),
        dcc.Interval(id=IDS["tick"], interval=500, disabled=True),

        html.Div(
            className="header",
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr auto",  # title grows, logo on the right
                "alignItems": "center",
                "gap": "12px",
                "padding": "12px 16px",
                "borderBottom": "1px solid #eaeaea",
                "position": "sticky",   # stays at top when scrolling
                "top": 0,
                "background": "white",
                "zIndex": 10,
            },
            children=[
                html.Div([
                    html.H1(
                        "Room Scan Flightpath",
                        style={"margin": 0, "fontSize": "24px", "letterSpacing": "0.3px"}
                    ),
                    html.Div(
                        "Flight Path generator for indoor environments",
                        style={"marginTop": "4px", "color": "#666", "fontSize": "14px"}
                    ),
                ]),
                html.A(
                    html.Img(
                        src=f"data:image/jpeg;base64,{logo_b64}",
                        style={"height": "40px", "width": "auto"},
                        alt="University Logo",
                    ),
                    href="#",  # put your university/site URL if you want it clickable
                    target="_blank",
                    style={"display": "inline-block"},
                ),
            ],
        ),

        html.Div(
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
                            options=[{"label": "2D (floor)", "value": "2D"}, {"label": "3D (stacked)", "value": "3D"}],
                            value="2D",
                            clearable=False,
                        ),
                        html.Hr(),
                        # --- Spiral / camera parameters ---
                    html.H3("Parameter Setup", style=TITLE_STYLE),

                    # Overlap slider (few marks so it's not cluttered)
                    dcc.Slider(
                        id="overlap",
                        min=0.0, max=0.9, step=0.05, value=0.3,
                        marks={0.0: "0.0", 0.3: "0.3", 0.6: "0.6", 0.9: "0.9"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Div("Overlap (0–0.9)", style={"fontSize": 12, "color": "#666"}),

                    # Lateral clearances (LABELLED)  ⬅ replaces the old unlabeled pair
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

                    # --- Spiral basics (labels + tooltips) ---
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

                                    # 3D-only fields (hidden unless scan-mode == "3D")
                    html.Div(
                        id="3d-only",
                        style={"display": "none", "marginTop": "6px"},
                        children=[
                            # --- fixed 3-column row ---
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)", "gap": "10px"},
                                children=[
                                    # Number of spiral layers
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

                                    # Vertical spacing (default / fallback)
                                    #html.Div(
                                    #    style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                    #    children=[
                                    #        html.Label(
                                    #            [
                                    #                "Δz per layer (m)",
                                    #                html.Span(
                                    #                    " ⓘ",
                                    #                    title="Meters between successive heights (vertical step). Used as default/fallback.",
                                    #                    style=TIP_ICON_STYLE,
                                    #                ),
                                    #            ],
                                    #            htmlFor="layer-height",
                                    #            style={"fontSize": 12, "color": "#555"},
                                    #        ),
                                    #        dcc.Input(id="layer-height", type="number", value=0.4, min=0.1, step=0.1, placeholder="e.g., 0.4"),
                                    #    ],
                                    #),

                                    # Optional top cap
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

                            # --- dynamic per-gap Δz inputs (one per gap) ---
                            html.Div(id="dz-container", style={"marginTop": "8px"}, children=[]),
                        ],
                    ),


                        html.Hr(),

                        html.Hr(),
                        html.Div(
                            style={"display": "flex", "gap": "8px"},
                            children=[
                                html.Button("Generate Path", id=IDS["btn_generate"], n_clicks=0),
                                html.Button("Reset", id=IDS["btn_reset"], n_clicks=0),
                            ],
                        ),
                        html.Div(id=IDS["error"], style={"color": "crimson", "fontSize": 13}),
                        html.Hr(),

                        # --- Mission summary card (filled by callback) ---
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

                # ----- Main panel -----
                html.Div(
                    className="main",
                    style={"minWidth": 0, "position": "relative", "overflow": "hidden"},
                    children=[
                        html.Div(
                            id="panel-wrap",
                            style={"width": "100%", "maxWidth": "900px", "marginLeft": "auto"},
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
                                            figure=INITIAL_FIG,  # define this above app.layout
                                            style={"height": "70vh", "width": "100%"},
                                        ),
                                        html.Div(
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "auto 1fr auto auto auto",
                                                "alignItems": "center",
                                                "gap": "8px",
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
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)
                                          


# ===================== Helpers =====================

def wps_to_df(wps: List[Waypoint]) -> pd.DataFrame:
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

    # Lock ranges to start at (0,0) and DO NOT square the aspect
    fig.update_xaxes(
        range=[0, L],
        autorange=False,
        fixedrange=True,
        constrain="range",
        tick0=0, dtick=1,
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
        tick0=0, dtick=1,
        zeroline=False,
        showline=True, mirror=True, linecolor="#444", linewidth=1,
        ticks="outside",
        # <-- key change: NO scaleanchor here
        title_text="y (m)"
    )

    fig.update_layout(
        template="plotly",
        height=600,
        margin=dict(l=40, r=20, t=20, b=16)  # tighter bottom margin
    )
    return fig







def make_3d_fig(df: pd.DataFrame, L: float, W: float, H: float, step_idx: int) -> go.Figure:
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

    # Big zoom-out multiplier
    zoom_out_factor = 1.0  # increase this for even more zoom-out
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
        height=600,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    return fig




def summaries_text(metrics: dict, df: pd.DataFrame) -> str:
    if metrics is None or df is None or df.empty:
        return ""
    return (
        f"Waypoints: {len(df)} | Total dist: {metrics.get('distance', 0):.1f} m | "
        f"ETA: {metrics.get('total_time', 0)/60:.1f} min | Battery: {metrics.get('battery_usage_percent', 0):.0f}%"
    )


def _nice_step(n: int, target_labels: int = 7) -> int:
    raw = max(1, int(math.ceil(n / max(target_labels, 2))))
    mag = 10 ** int(math.floor(math.log10(raw)))
    for m in (1, 2, 5, 10):
        s = m * mag
        if s >= raw:
            return s
    return raw

def make_slider_marks_from_vmax(vmax: int) -> dict:
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


def _fmt_sec(s: float) -> str:
    # show sec for small values, minutes for larger; avoids huge decimals
    if s < 120:
        return f"{s:.1f} s"
    return f"{s/60:.1f} min"

def render_metrics_card(_metrics: dict, df: pd.DataFrame, mode: str) -> html.Div:
    """
    Strict summary that matches the original (old) validator logic.
    - 2D: cruise = sum 2D leg lengths; turns counted by ~90° between consecutive 2D legs.
    - 3D: total distance sums 3D legs; in-layer cruise uses 2D; seams add 3D distance and (2D/v_horiz + |dz|/v_vert) time.
    - Pauses: controller_wp_pause_s * len(waypoints). Fixed overhead = 12.0 s.
    - Hover: sum of waypoint hold_time.
    - t90 from power_rot=46.67, base_power=60, base_yaw_deg_s=120.
    """
    # Early empty UI
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

    # ---------- constants (exactly as in old code) ----------
    drone = DEFAULT_DRONE
    include_takeoff_landing = True
    ground_z = 0.0
    controller_wp_pause_s = 2.0       # set_waypoint_pause(2.0)
    fixed_pre_post_pauses_s = 12.0     # pause(4)+pause(1)+pause(1)+pause(6)

    v_horiz = drone.speed
    v_vert = getattr(drone, "vertical_speed", None) or drone.speed

    base_power = 60.0
    base_yaw_deg_s = 120.0
    power_rot = 46.67
    yaw_rate_large = base_yaw_deg_s * (power_rot / base_power)
    t90 = 90.0 / max(yaw_rate_large, 1e-6)

    eps_z = 1e-9

    # ---------- decide 2D vs 3D by contiguous Z changes ----------
    z_vals = [wp.z for wp in wps]
    z_runs = 1
    for a, b in zip(z_vals, z_vals[1:]):
        if abs(a - b) > eps_z:
            z_runs += 1
    use_3d = (mode == "3D") or (z_runs > 1)

    # ---------- compute (exact old logic) ----------
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
        # Build layers (contiguous runs of ~equal Z)
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

    # ---------- render (same card layout you use) ----------
    n_wps_display = len(df)  # old print used "Total waypoints" = len(waypoints)
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
            _badge(bool(metrics["battery_ok"]), "OK", "Low"),
            html.Span("  •  Warning: "),
            _badge(not bool(metrics["battery_warning"]), "No", "Yes"),
            html.Span("  •  Distance: "),
            _badge(bool(metrics["distance_ok"]), "OK", "Limit Exceeded"),
        ],
        style={"gridColumn": "1 / -1", "display": "flex", "gap": "8px", "alignItems": "center"},
    )

    return html.Div(
        [html.Strong("Mission summary"), _grid(primary_facts), _grid(secondary_facts), breakdown_line, status_line],
        style={"display": "grid", "rowGap": "8px"},
    )






# ===================== Callbacks =====================
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
    #State("layer-height", "value"),
    State("top-cap", "value"),
    State({"type": "dz", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def generate(nc, L, W, H, mode, overlap, wall_off, clearance, alt, turns, layers, z_cap, dz_list):
    try:
        # Input validation
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
            feasible, metrics = validate_mission(drone, wps)
            df = wps_to_df(wps)
            fig = make_2d_fig(df, L, W, 0 if df.empty else int(df["step"].max()))
        else:
            # 3D mode with ABSOLUTE heights per layer (from dz_list)
            base = generate_spiral_scan(
                drone, dims, float(alt), float(overlap),
                float(wall_off or 0), float(clearance or 0), int(turns or 3)
            )

            n_layers = int(layers or 1)
            if n_layers < 1:
                n_layers = 1

            # Build absolute heights from inputs; fallback to 'alt' if missing/invalid
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
                # Enforce drone minimum altitude
                h = max(drone.min_altitude, h)
                heights.append(h)

            # Optional top cap
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
            feasible, metrics = validate_mission_with_layers(drone, wps)
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
    if vmax is None:
        return no_update
    val = int(val or 0)
    vmax = int(vmax or 0)
    return 0 if val >= vmax else val + 1


# ---------------- Downloads (in-memory) ----------------

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


# ---------------- Run (local dev) ----------------
# For hosted PaaS, use: gunicorn app:server
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)