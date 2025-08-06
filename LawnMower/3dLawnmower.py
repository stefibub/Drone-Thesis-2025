# Re-running the code with the missing DroneConfig definition and required imports
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os 
import csv 
from datetime import datetime
import matplotlib.pyplot as plt

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
    turning_radius: float
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

def add_center_line_scan(
    x_fixed: float, y_start: float, y_end: float, z: float,
    reverse: bool, speed: float, hold: float, spacing: float
    ) -> List[Waypoint]:
    if reverse:
        y_start, y_end = y_end, y_start
    dy = y_end - y_start
    n_points = max(1, math.ceil(abs(dy) / spacing))
    step = dy / n_points

    return [
        Waypoint(x_fixed, y_start + i * step, z, 0.0, speed, hold)
        for i in range(n_points + 1)
        ]


def sample_diagonal(
    start: Waypoint,
    end: Waypoint,
    spacing: float,
    speed: float,
    hold: float,
    endpoints_only: bool = False
) -> List[Waypoint]:
    """
    Returns waypoints from start→end.
    - If endpoints_only=True, returns just the end waypoint (start already in all_wps).
    - Otherwise returns evenly spaced points including end.
    """
    if endpoints_only:
        return [Waypoint(
            x=end.x,
            y=end.y,
            z=end.z,
            gimbal_pitch=0.0,
            speed=speed,
            hold_time=hold
        )]

    dx, dy, dz = end.x - start.x, end.y - start.y, end.z - start.z
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    raw_steps = math.ceil(dist / spacing)
    n_steps = max(2, raw_steps)
    return [
        Waypoint(
            x = start.x + dx * (i / n_steps),
            y = start.y + dy * (i / n_steps),
            z = start.z + dz * (i / n_steps),
            gimbal_pitch = 0.0,
            speed = speed,
            hold_time = hold
        )
        for i in range(1, n_steps + 1)
    ]


def generate_wall_with_center_scan(
        drone: DroneConfig,
        dims: Tuple[float, float, float],
        overlap: float,
        wall_offset: float,
        clearance: float = 0.0,
        floor_z_override: Optional[float] = None,
        ceiling_z_override: Optional[float] = None,
        min_vertical_slices: int = 2,
        exclude_endpoints: bool = False,
        endpoints_only_center_scan: bool = False,
        endpoints_only_diagonal: bool = False,
        endpoints_only_sweep:bool = False 
) -> List[Waypoint]:
    w, l, h = dims
    fp_w_wall, fp_h_wall = calculate_footprint(drone, wall_offset)
    along_wall_spacing = fp_w_wall * (1 - overlap)
    vertical_band_spacing = fp_h_wall * (1 - overlap)
    speed_along = calculate_speed(fp_w_wall, overlap, drone.fps)
    hold = 1.0 / drone.fps + drone.hover_buffer

    default_floor_z = max(clearance, drone.min_altitude)
    default_ceiling_z = max(h - clearance, drone.min_altitude)
    floor_z = floor_z_override if floor_z_override is not None else default_floor_z
    ceiling_z = ceiling_z_override if ceiling_z_override is not None else default_ceiling_z
    height_range = ceiling_z - floor_z
    if height_range <= 0:
        return []

    raw_nbands = math.ceil(height_range / vertical_band_spacing) if fp_h_wall > 0 else 1
    nbands = max(min_vertical_slices, raw_nbands)
    band_height = height_range / nbands

    def make_sweep(start: Tuple[float, float], end: Tuple[float, float], z_val: float, forward: bool) -> List[Waypoint]:
        if not forward:
            start, end = end, start

        x0, y0 = start
        x1, y1 = end

        if math.isclose(x0, x1, abs_tol=1e-6):
            dy = y1 - y0
            dist = abs(dy)
            n_samples = max(1, math.ceil(dist / along_wall_spacing))
            delta_y = dy / n_samples
            return [Waypoint(x0, y0 + i * delta_y, z_val, 0.0, speed_along, hold) for i in range(n_samples + 1)]

        elif math.isclose(y0, y1, abs_tol=1e-6):
            dx = x1 - x0
            dist = abs(dx)
            n_samples = max(1, math.ceil(dist / along_wall_spacing))
            delta_x = dx / n_samples
            return [Waypoint(x0 + i * delta_x, y0, z_val, 0.0, speed_along, hold) for i in range(n_samples + 1)]

        else:
            raise ValueError(f"Sweep from {start} to {end} is diagonal, which is not allowed.")

    all_wps: List[Waypoint] = []
    walls = [
        ((wall_offset, wall_offset), (w - wall_offset, wall_offset)),            # +x
        ((w - wall_offset, wall_offset), (w - wall_offset, l - wall_offset)),    # +y
        ((w - wall_offset, l - wall_offset), (wall_offset, l - wall_offset)),    # -x
        ((wall_offset, l - wall_offset), (wall_offset, wall_offset)),            # -y
    ]

    for wall_idx, (wall_start, wall_end) in enumerate(walls):
        for band_idx in range(nbands + 1):
            if exclude_endpoints and (band_idx == 0 or band_idx == nbands):
                continue
            z_val = floor_z + band_idx * band_height
            z_val = max(z_val, drone.min_altitude)
            forward = (band_idx % 2 == 0)
            sweep = make_sweep(wall_start, wall_end, z_val, forward)

            if endpoints_only_sweep and len(sweep) >= 2: 
                sweep = [sweep[0], sweep[-1]]

            if all_wps and sweep:
                last_wp = all_wps[-1]
                next_wp = sweep[0]
                if not math.isclose(last_wp.x, next_wp.x, abs_tol=1e-6):
                    all_wps.append(Waypoint(next_wp.x, last_wp.y, last_wp.z, 0.0, drone.speed, hold))
                if not math.isclose(last_wp.y, next_wp.y, abs_tol=1e-6):
                    all_wps.append(Waypoint(next_wp.x, next_wp.y, last_wp.z, 0.0, drone.speed, hold))
                if not math.isclose(last_wp.z, next_wp.z, abs_tol=1e-6):
                    all_wps.append(Waypoint(next_wp.x, next_wp.y, next_wp.z, 0.0, drone.speed, hold))
            
            all_wps.extend(sweep)

            if wall_idx == 1:  # After +y wall, insert center scan
                center_x = (wall_offset + (w - wall_offset)) / 2
                center_line = add_center_line_scan(
                    x_fixed    = center_x,
                    y_start    = wall_offset,
                    y_end      = l - wall_offset,
                    z          = z_val,
                    reverse    = not forward,
                    speed      = speed_along,
                    hold       = hold,
                    spacing    = along_wall_spacing
                )
                # optionally keep only endpoints
                if endpoints_only_center_scan and len(center_line) >= 2:
                    center_line = [center_line[0], center_line[-1]]
                all_wps.extend(center_line)

                if center_line:
                    diag_start_wp = center_line[-1]
                    # compute opposite corner
                    if math.isclose(diag_start_wp.y, wall_offset, abs_tol=1e-6):
                        target_corner_y = l - wall_offset
                    else:
                        target_corner_y = wall_offset
                    target_corner_x = (w - wall_offset) if center_x < (w/2) else wall_offset
                    diag_end_wp = Waypoint(
                        x = target_corner_x,
                        y = target_corner_y,
                        z = z_val,
                        gimbal_pitch = 0.0,
                        speed = speed_along,
                        hold_time = hold
                    )

                    diagonal_wps = sample_diagonal(
                        start          = diag_start_wp,
                        end            = diag_end_wp,
                        spacing        = along_wall_spacing,
                        speed          = speed_along,
                        hold           = hold,
                        endpoints_only = endpoints_only_diagonal
                    )
                    #print(f"Generated {len(diagonal_wps)} diagonal waypoints at z={z_val:.2f}")
                    all_wps.extend(diagonal_wps)

    return all_wps

def filter_duplicates(wps: List[Waypoint], tol: float = 1e-6) -> List[Waypoint]:
    """
    Return a new list of waypoints where consecutive duplicates
    (within tol in x,y,z) have been removed.
     """
    if not wps:
        return []

    filtered = [wps[0]]
    for wp in wps[1:]:
        prev = filtered[-1]
        dx = abs(wp.x - prev.x)
        dy = abs(wp.y - prev.y)
        dz = abs(wp.z - prev.z)
        if dx > tol or dy > tol or dz > tol:
            filtered.append(wp)
    return filtered

def reduce_waypoints_collinear(waypoints: List[Waypoint], eps=1e-6) -> List[Waypoint]:
    if len(waypoints) <= 2:
        return waypoints[:]
    reduced = [waypoints[0]]
    for prev, curr, nxt in zip(waypoints, waypoints[1:], waypoints[2:]):
        # build vectors prev→curr and curr→next
        v1 = (curr.x - prev.x, curr.y - prev.y, curr.z - prev.z)
        v2 = (nxt.x - curr.x, nxt.y - curr.y, nxt.z - curr.z)
        # cross‐product to test collinearity in 3D
        cross = (
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0],
        )
        if math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2) < eps:
            # still going straight – drop `curr`
            continue
        reduced.append(curr)
    reduced.append(waypoints[-1])
    return reduced

def visualize_waypoints_3d(
    waypoints: List[Waypoint],
    elev: float = 30,
    azim: float = 45,
    draw_lines: bool = True,
    label_every: int = 1,
    label_offset: Tuple[float, float, float] = (0.03, 0.03, 0.02),
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None,
    text_kwargs: Optional[Dict] = None
):
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]

    line_kwargs = line_kwargs or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.8}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.9}
    text_kwargs = text_kwargs or {'size': 8, 'zorder': 2}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if draw_lines:
        ax.plot(xs, ys, zs, **line_kwargs)
    ax.scatter(xs, ys, zs, **marker_kwargs)

    base_offset_x, base_offset_y, base_offset_z = label_offset
    jitter_scale = 0.03  # scale for offset spacing

    #for idx, (x, y, z) in enumerate(zip(xs, ys, zs), start=1):
    #    if (idx - 1) % label_every != 0:
    #        continue

        # Apply consistent, patterned jitter to avoid overlap
    #    jitter_x = base_offset_x + ((-1)**idx) * jitter_scale * (idx % 3)
    #    jitter_y = base_offset_y + ((-1)**(idx + 1)) * jitter_scale * ((idx + 1) % 3)
    #    jitter_z = base_offset_z + (idx % 2) * jitter_scale * 0.5

    #    ax.text(
    #        x + jitter_x,
    #        y + jitter_y,
    #        z + jitter_z,
    #        str(idx),
    #        **text_kwargs
    #    )

    ax.set(title='3D Flightpath', xlabel='x', ylabel='y', zlabel='z')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()




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

def validate_mission(drone: DroneConfig, waypoints: List[Waypoint]) -> Tuple[bool, dict]:
    total_distance = 0.0
    total_travel_time = 0.0

    for wp1, wp2 in zip(waypoints, waypoints[1:]):
        dx, dy, dz = wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        speed = drone.speed
        total_distance += dist
        total_travel_time += dist / speed

    last_wp = waypoints[-1]
    return_dist = math.sqrt(last_wp.x**2 + last_wp.y**2 + last_wp.z**2)
    return_time = return_dist / (last_wp.speed or drone.speed)
    total_distance += return_dist
    total_travel_time += return_time

    total_hover_time = sum(wp.hold_time for wp in waypoints)
    total_time = total_travel_time + total_hover_time

    battery_used_pct = (total_time / drone.max_battery_time) * 100
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



if __name__ == '__main__':
    cfg = DroneConfig(
        weight=0.292,
        max_battery_time=1380.0,
        max_distance=5000.0,
        horizontal_fov=82.1,
        vertical_fov=52.3,
        fps=60.0,
        resolution=(2720, 1530),
        speed=0.4,
        min_altitude=1.0,
        turning_radius=1.0,
        hover_buffer=10,
        battery_warning_threshold=0.85
    )
    dims = (5.0, 5.0, 2.2)
    overlap = 0.7
    wall_offset = 1.0
    clearance = 0.1

    floor_z = 1.5
    ceiling_z = max(dims[2] - clearance, cfg.min_altitude)
    epsilon = 0.5
    ceiling_z = ceiling_z - epsilon

    raw_waypoints = generate_wall_with_center_scan(
        cfg,
        dims,
        overlap,
        wall_offset,
        clearance,
        floor_z_override=floor_z,
        ceiling_z_override=ceiling_z,
        min_vertical_slices=4,
        exclude_endpoints=True,
        endpoints_only_diagonal= True,
        endpoints_only_sweep= True,
        endpoints_only_center_scan= True
    )
    #reduced_waypoints = reduce_waypoints_collinear(raw_waypoints)
    waypoints = filter_duplicates(raw_waypoints)
   
    feasible, metrics = validate_mission(cfg, waypoints)

    out_dir = "3DLawnmower_FlightPlanData"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"mission_path_lawnmower_3d_{timestamp}"

    def altitude_range(waypoints: List[Waypoint]) -> Tuple[float, float]:
        zs = [wp.z for wp in waypoints]
        return min(zs), max(zs)

    z_min, z_max = altitude_range(waypoints)
    print(f"Wall-only scan: {len(waypoints)} waypoints, feasible: {feasible}")
    print(f"Altitude span: {z_min:.2f} to {z_max:.2f} m (expected between {floor_z} and {ceiling_z})")
    print(f"Distance: {metrics['distance']:.1f} m, Total time: {metrics['total_time']/60:.1f} mins")

    csv_filename = os.path.join(out_dir, f"{base_name}.csv")
    txt_filename = os.path.join(out_dir, f"{base_name}.txt")
    dpt_filename = os.path.join(out_dir, f"{base_name}.dpt")

    # Export
    export_to_marvelmind(waypoints, csv_filename)
    export_to_marvelmind(waypoints, txt_filename)
    export_to_dpt(waypoints, cfg, dpt_filename)

    print(f"Export Complete:\n{csv_filename},\n{txt_filename},\n{dpt_filename}.\n")
    visualize_waypoints_3d(waypoints, draw_lines=True)


    #visualize_waypoints_3d(
    #    waypoints,
    #    label_every=0,
    #   label_offset=(0, 0, 0),
    #    text_kwargs={'size': 7, 'color': 'red'}
    #)