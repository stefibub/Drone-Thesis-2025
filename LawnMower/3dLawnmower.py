import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt


@dataclass
class DroneConfig:
    """
    holds drone settings and limits
    """
    weight: float  # weight in kg
    max_battery_time: float  # max flight time in seconds - tested
    max_distance: float  # max travel distance in meters
    horizontal_fov: float  # camera horizontal field of view in degrees
    vertical_fov: float  # camera vertical field of view in degrees
    fps: float  # camera frame rate in frames per second
    resolution: Tuple[int, int]  # camera resolution in pixels (width, height)
    speed: float  # nominal flight speed in m/s
    min_altitude: float  # minimum safe flight altitude in meters
    turning_radius: float  # minimum turn radius in meters
    hover_buffer: float  # extra hover time for stabilization in seconds
    battery_warning_threshold: float  # warning if battery % used exceeds this


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


def generate_planar_scan(
        span_long: float,
        span_short: float,
        spacing: float,
        z: float,
        pitch: float,
        speed: float,
        offset: float,
        is_x_aligned: bool,
        turning_radius: float,
        hold_time: float
) -> List[Waypoint]:
    n_passes = max(1, math.ceil(span_short / spacing))
    delta_short = span_short / n_passes
    waypoints: List[Waypoint] = []
    for i in range(n_passes + 1):
        coord_short = offset + i * delta_short
        if i % 2 == 0:
            start_long = offset + turning_radius
            end_long = offset + span_long - turning_radius
        else:
            start_long = offset + span_long - turning_radius
            end_long = offset + turning_radius
        seg_len = abs(end_long - start_long)
        n_samples = max(1, math.ceil(seg_len / spacing))
        delta_long = (end_long - start_long) / n_samples
        for j in range(n_samples + 1):
            pos_long = start_long + j * delta_long
            x = pos_long if is_x_aligned else coord_short
            y = coord_short if is_x_aligned else pos_long
            waypoints.append(Waypoint(x, y, z, pitch, speed, hold_time))
    return waypoints


def generate_cube_scan(
        drone: DroneConfig,
        dims: Tuple[float, float, float],
        overlap: float,
        wall_offset: float,
        clearance: float = 0.0
) -> List[Waypoint]:
    w, l, h = dims
    fp_w, fp_h = calculate_footprint(drone, h)
    spacing_x = fp_w * (1 - overlap)
    spacing_y = fp_h * (1 - overlap)
    speed_xy = calculate_speed(fp_w, overlap, drone.fps)
    hold = 1.0 / drone.fps + drone.hover_buffer
    inset = wall_offset + clearance
    all_wps: List[Waypoint] = []

    # Top and bottom planar scans
    all_wps += generate_planar_scan(
        span_long=w - 2 * inset,
        span_short=l - 2 * inset,
        spacing=spacing_x,
        z=max(clearance, drone.min_altitude),
        pitch=-90.0,
        speed=speed_xy,
        offset=inset,
        is_x_aligned=True,
        turning_radius=drone.turning_radius,
        hold_time=hold
    )
    all_wps += generate_planar_scan(
        span_long=l - 2 * inset,
        span_short=w - 2 * inset,
        spacing=spacing_y,
        z=max(h - clearance, drone.min_altitude),
        pitch=60.0,
        speed=speed_xy,
        offset=inset,
        is_x_aligned=False,
        turning_radius=drone.turning_radius,
        hold_time=hold
    )

    # Vertical wall scans
    fp_w_wall, fp_h_wall = calculate_footprint(drone, wall_offset)
    ss = fp_w_wall * (1 - overlap)
    sh = fp_h_wall * (1 - overlap)
    speed_z = calculate_speed(fp_h_wall, overlap, drone.fps)
    for axis, pos in [('y', wall_offset), ('y', l - wall_offset),
                      ('x', wall_offset), ('x', w - wall_offset)]:
        span = w if axis == 'y' else l
        height_range = h - 2 * clearance
        nspan = max(1, math.ceil((span - 2 * wall_offset) / ss))
        nheight = max(1, math.ceil(height_range / sh))
        ds = (span - 2 * wall_offset) / nspan
        dh = height_range / nheight
        for i in range(nspan + 1):
            p_span = i * ds + wall_offset
            rows = (range(nheight + 1) if i % 2 == 0 else range(nheight, -1, -1))
            for j in rows:
                z_pt = j * dh + clearance
                x = p_span if axis == 'y' else pos
                y = pos if axis == 'y' else p_span
                all_wps.append(Waypoint(x, y, max(z_pt, drone.min_altitude), 0.0, speed_z, hold))
    return all_wps


def validate_mission(drone: DroneConfig, waypoints: List[Waypoint]) -> Tuple[bool, dict]:
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


def visualize_waypoints_3d(
    waypoints: List[Waypoint],
    elev: float = 30,
    azim: float = 45,
    draw_lines: bool = True,
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None
):
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    line_kwargs = line_kwargs or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.8}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.9}

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    if draw_lines:
        ax.plot(xs, ys, zs, **line_kwargs)
    ax.scatter(xs, ys, zs, **marker_kwargs)
    ax.set(title='3D Flightpath', xlabel='x', ylabel='y', zlabel='z')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    cfg = DroneConfig(
        weight=0.292,
        max_battery_time=1380.0,
        max_distance=5000.0,
        horizontal_fov=82.1,
        vertical_fov=52.3,
        fps=60.0,
        resolution=(2720, 1530),
        speed=0.5,
        min_altitude=1.0,
        turning_radius=1.0,
        hover_buffer=10,
        battery_warning_threshold=0.85
    )
    dims = (6.0, 6.0, 3.0)
    overlap = 0.7
    wall_offset = 2.0
    clearance = 0.75

    waypoints = generate_cube_scan(cfg, dims, overlap, wall_offset, clearance)
    feasible, metrics = validate_mission(cfg, waypoints)
    print(f"3D scan: {len(waypoints)} waypoints, feasible: {feasible}")
    print(f"Distance: {metrics['distance']:.1f} m, Total time: {metrics['total_time']/60:.1f} mins")

    visualize_waypoints_3d(waypoints)
