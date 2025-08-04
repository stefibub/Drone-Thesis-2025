import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime


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

    The spiral starts at the center of the inset rectangle (accounting for wall_offset + clearance)
    and expands outward in the order: right, up, left, down, with layer lengths increasing by one
    every two directions. The step size is chosen to respect the desired camera coverage (footprint &
    overlap) but is also constrained so that the spiral produces approximately `target_turns` concentric
    layers before hitting the smallest span of the scan area.
    """
    inset = wall_offset + clearance
    hold = 1.0 / drone.fps + drone.hover_buffer

    min_x = inset
    max_x = dims[0] - inset
    min_y = inset
    max_y = dims[1] - inset

    if min_x >= max_x or min_y >= max_y:
        return []

    altitude = max(clearance, drone.min_altitude)

    # nominal footprint-based spacing (for camera coverage) but override with tighter step
    fp_long, fp_short = calculate_footprint(drone, altitude)
    avg_fp = (fp_long + fp_short) / 2.0
    nominal_step = avg_fp * (1 - overlap)
    speed = calculate_speed(avg_fp, overlap, drone.fps)

    # deciding on the step so that the spiral has roughly target_turns layers before hitting the smallest span
    span_x = max_x - min_x
    span_y = max_y - min_y
    min_span = min(span_x, span_y)

    # each "full turn" increases in both dimensions by 2*step (right+up+left+down),
    # so to get target_turns from center to edge, step ≈ min_span / (2*target_turns + 1)
    step = min(nominal_step, min_span / (2 * target_turns + 1))

    # note: if nominal_step is already smaller, use it (you can also force tighter by ignoring nominal_step)
    # step = min_span / (2 * target_turns + 1)  # uncomment to force exact density regardless of footprint

    # start at center
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    x, y = cx, cy

    wps: List[Waypoint] = [Waypoint(x, y, altitude, 0.0, speed, hold)]

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, up, left, down
    layer = 1
    dir_idx = 0

    #  outward spiral until next step would exit
    while True:
        moved = False
        for _ in range(2):  # two directions per layer count
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
    """
    removing intermediate waypoints that lie collinearly on straight segments.

    logic: preserves the first and last waypoint. For any triplet (prev, curr, next), if curr lies
    exactly on the same line and direction as prev→next (within tolerance), it is skipped,
    reducing redundancy along straight legs.
    returns: reduced list of waypoints with unnecessary intermediate points removed.
    """
    if len(waypoints) <= 2:
        return waypoints.copy()
    reduced = [waypoints[0]]
    for prev, curr, nxt in zip(waypoints, waypoints[1:], waypoints[2:]):
        v1 = (curr.x - prev.x, curr.y - prev.y, curr.z - prev.z)
        v2 = (nxt.x - curr.x, nxt.y - curr.y, nxt.z - curr.z)
        # cross product to test collinearity
        cross = (v1[1]*v2[2] - v1[2]*v2[1],
                 v1[2]*v2[0] - v1[0]*v2[2],
                 v1[0]*v2[1] - v1[1]*v2[0])
        cross_norm = math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        # keep curr if direction changes (non-collinear) or reverses
        if cross_norm < eps and dot > 0:
            continue  # redundant straight-point
        reduced.append(curr)
    reduced.append(waypoints[-1])
    return reduced


def visualize_waypoints_2d(
    waypoints: List[Waypoint],
    draw_lines: bool = False,
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None
):
    """
    plot top-down, side, and front views of waypoints in 2d
    draw_lines: connect points if true
    """
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    line_kwargs = line_kwargs or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.7, 'color': 'C2'}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.8, 'color': 'C1'}

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (X, Y, title) in zip(
        axs,
        [(xs, ys, 'top-down'), (xs, zs, 'side'), (ys, zs, 'front')]
    ):
        if draw_lines:
            ax.plot(X, Y, **line_kwargs)
        ax.scatter(X, Y, **marker_kwargs)
        ax.set(title=title,
               xlabel='x' if title != 'front' else 'y',
               ylabel='y' if title == 'top-down' else 'z')
        ax.grid(True)
        if title == 'top-down':
            ax.axis('equal')

    plt.tight_layout()
    plt.show()


##### Mission Validation #####


def validate_mission(drone: DroneConfig, waypoints: List[Waypoint]) -> Tuple[bool, dict]:
    """
    Evaluate mission feasibility using real-world tested max battery time.
    Calculates travel time and hover time, then checks if the mission fits within limits.
    Returns feasibility as boolean and dictionary of metrics.
    """
    total_distance = 0.0
    total_travel_time = 0.0

    for wp1, wp2 in zip(waypoints, waypoints[1:]):
        dx, dy, dz = wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        speed = drone.speed
        total_distance += dist
        total_travel_time += dist / speed

    # return to origin
    last_wp = waypoints[-1]
    return_dist = math.sqrt(last_wp.x**2 + last_wp.y**2 + last_wp.z**2)
    return_time = return_dist / (last_wp.speed or drone.speed)
    total_distance += return_dist
    total_travel_time += return_time

    # sums all the hover times (holding time at each waypoint)
    total_hover_time = sum(wp.hold_time for wp in waypoints)
    total_time = total_travel_time + total_hover_time

    # calculate battery usage as a percentage of the max tested flight time
    battery_used_pct = (total_time / drone.max_battery_time) * 100
    # check if mission fits within battery time and distance limits
    battery_ok = battery_used_pct <= 100.0
    battery_warning = battery_used_pct >= drone.battery_warning_threshold * 100
    distance_ok = total_distance <= drone.max_distance
    feasible = battery_ok and distance_ok

    return feasible, {
        'distance': total_distance,
        'travel_time': total_travel_time,
        'hover_time': total_hover_time,
        'total_time': total_time,
        'battery_usage_percent': battery_used_pct,
        'battery_ok': battery_ok,
        'battery_warning': battery_warning,
        'distance_ok': distance_ok
    }


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
    lines.append(f"set_power_rot(60.00)")
    lines.append(f"set_power_height(10.00)")
    lines.append(f"set_waypoint_pause(2.0)") # Fixed value as requested in prior turn
    lines.append(f"set_timeout(9.0)")
    lines.append(f"set_scan_timeout(10.0)")
    lines.append(f"set_waypoint_radius(0.10)")
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

    # Write all lines to the .dpt file
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    cfg = DroneConfig(
        weight=0.292,
        max_battery_time=1380.0,
        horizontal_fov=82.1,
        vertical_fov=52.3,
        fps=60.0,
        resolution=(2720, 1530),
        speed=0.5,
        min_altitude=1.0,
        hover_buffer=10,
        battery_warning_threshold=0.85,
        max_distance = 5000.0
    )
    dims = (6.0, 6.0, 3.0)
    overlap = 0.8
    wall_offset = 2.0
    clearance = 0.75

    out_dir = "2DSpiral_FlightPlanData"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"mission_path_spiral_2d_{timestamp}"

    floor_altitude = cfg.min_altitude
    ceiling_altitude = max(dims[2] - clearance, cfg.min_altitude)

    print(f"\n--- 2D Scan with Planar Spiral Pattern ---\n")
    wps = generate_spiral_scan(
        drone=cfg,
        dims=dims,
        overlap=overlap,
        wall_offset=wall_offset,
        clearance=clearance,
        altitude=floor_altitude
    )
    wps_reduced = reduce_waypoints_collinear(wps)
    wps = wps_reduced
    feasible, metrics = validate_mission(cfg, wps)
    print(f"Total waypoints: {len(wps)}")

    print(f"Distance: {metrics['distance']:.1f} m,\n"
    f"Total time: {metrics['total_time']/60:.1f} mins,\n"
    f"Feasible: {feasible}\n")

    # consistent filenames
    csv_filename = os.path.join(out_dir, f"{base_name}.csv")
    txt_filename = os.path.join(out_dir, f"{base_name}.txt")
    dpt_filename = os.path.join(out_dir, f"{base_name}.dpt")

    # Export
    export_to_marvelmind(wps, csv_filename)
    export_to_marvelmind(wps, txt_filename)
    export_to_dpt(wps, cfg, dpt_filename)

    print(f"Export Complete:\n{csv_filename},\n{txt_filename},\n{dpt_filename}.\n")
    visualize_waypoints_2d(wps, draw_lines=True)
