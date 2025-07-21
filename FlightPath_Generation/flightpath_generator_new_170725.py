"""
optimized cube scan module

this module generates waypoints to scan all faces of a cuboid (floor, ceiling, walls) using a drone camera
"""

import math  # trigonometry and distance calculations
import csv   # write waypoints to csv file
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt  # plotting in 2d
from mpl_toolkits.mplot3d import Axes3D  # plotting in 3d


@dataclass
class DroneConfig:
    """
    holds drone performance limits and camera parameters
    """
    battery_time: float        # available flight time in seconds
    max_distance: float        # maximum travel distance in meters
    max_flight_time: float     # maximum mission duration in seconds
    horizontal_fov: float      # camera horizontal field-of-view (degrees)
    vertical_fov: float        # camera vertical field-of-view (degrees)
    fps: float                 # frames per second of camera
    resolution: Tuple[int,int] # (width, height) pixels
    speed: float               # nominal flight speed (m/s)
    min_altitude: float        # minimum safe altitude (m)


@dataclass
class Waypoint:
    """
    represents a single waypoint in 3d space with camera settings
    """
    x: float                   # x coordinate (m)
    y: float                   # y coordinate (m)
    z: float                   # altitude (m)
    gimbal_pitch: float        # camera pitch angle (degrees)
    speed: float               # flight speed at this waypoint (m/s)


def calculate_footprint(drone: DroneConfig, distance: float) -> Tuple[float, float]:
    """
    calculate camera footprint width and height at given distance

    uses pinhole camera model:
      width  = 2 * distance * tan(hfov/2)
      height = 2 * distance * tan(vfov/2)

    returns:
      (footprint_width, footprint_height) in meters
    """
    hfov_rad = math.radians(drone.horizontal_fov)
    vfov_rad = math.radians(drone.vertical_fov)
    width = 2 * distance * math.tan(hfov_rad / 2)
    height = 2 * distance * math.tan(vfov_rad / 2)
    return width, height


def calculate_speed(footprint_len: float, overlap: float, fps: float) -> float:
    """
    compute max flight speed to maintain image overlap

    step = footprint_len * (1 - overlap)
    speed = step * fps
    returns speed in m/s
    """
    return footprint_len * (1 - overlap) * fps


def generate_grid_waypoints(
    span_x: float,
    span_y: float,
    spacing_x: float,
    spacing_y: float,
    z: float,
    pitch: float,
    speed: float
) -> List[Waypoint]:
    """
    generate zig-zag grid of waypoints over a rectangle at altitude z

    - span_x, span_y: dimensions of area
    - spacing_x, spacing_y: desired overlap spacing
    - z: flight altitude
    - pitch: camera pitch angle
    - speed: flight speed

    returns list of waypoints covering the grid
    """
    nx = max(1, math.ceil(span_x / spacing_x))
    ny = max(1, math.ceil(span_y / spacing_y))
    dx = span_x / nx
    dy = span_y / ny

    waypoints: List[Waypoint] = []
    for row in range(ny + 1):
        y = row * dy
        cols = range(nx + 1) if row % 2 == 0 else range(nx, -1, -1)
        for col in cols:
            x = col * dx
            waypoints.append(Waypoint(x, y, z, pitch, speed))
    return waypoints


def generate_wall_waypoints(
    span: float,
    height_range: float,
    spacing_span: float,
    spacing_height: float,
    wall_offset: float,
    fixed_axis: str,
    pitch: float,
    speed: float
) -> List[Waypoint]:
    """
    generate up/down sweeps along a wall face

    - span: length of wall
    - height_range: vertical extent excluding clearances
    - spacing_span, spacing_height: desired spacing
    - wall_offset: distance from origin along fixed axis
    - fixed_axis: 'x' or 'y' for wall orientation
    - pitch, speed: camera and flight settings

    returns list of wall waypoints before clearance offset
    """
    nspan = max(1, math.ceil(span / spacing_span))
    nheight = max(1, math.ceil(height_range / spacing_height))
    ds = span / nspan
    dh = height_range / nheight

    waypoints: List[Waypoint] = []
    for i in range(nspan + 1):
        pos_span = i * ds
        rows = range(nheight + 1) if i % 2 == 0 else range(nheight, -1, -1)
        for j in rows:
            z = j * dh
            if fixed_axis == 'x':
                x, y = wall_offset, pos_span
            else:
                x, y = pos_span, wall_offset
            waypoints.append(Waypoint(x, y, z, pitch, speed))
    return waypoints


def generate_cube_scan(
    drone: DroneConfig,
    dims: Tuple[float, float, float],
    overlap: float,
    wall_offset: float,
    clearance: float = 0.0
) -> List[Waypoint]:
    """
    combine scans of floor, ceiling, and walls for full cuboid coverage

    - dims: (width, length, height) of box
    - overlap: image overlap fraction
    - wall_offset: lateral offset from walls
    - clearance: inward margin from surfaces
    """
    w, l, h = dims

    fp_w, fp_h = calculate_footprint(drone, h)
    sx, sy = fp_w * (1 - overlap), fp_h * (1 - overlap)
    speed_xy = calculate_speed(fp_w, overlap, drone.fps)

    all_wps: List[Waypoint] = []

    z_floor = max(clearance, drone.min_altitude)
    floor = generate_grid_waypoints(
        w - 2*clearance, l - 2*clearance,
        sx, sy, z_floor, -90.0, speed_xy
    )
    for wp in floor:
        wp.x += clearance
        wp.y += clearance
    all_wps.extend(floor)

    z_ceil = max(h - clearance, drone.min_altitude)
    ceiling = generate_grid_waypoints(
        w - 2*clearance, l - 2*clearance,
        sx, sy, z_ceil, 60.0, speed_xy
    )
    for wp in ceiling:
        wp.x += clearance
        wp.y += clearance
    all_wps.extend(ceiling)

    fp_w_wall, fp_h_wall = calculate_footprint(drone, wall_offset)
    ss, sh = fp_w_wall * (1 - overlap), fp_h_wall * (1 - overlap)
    speed_z = calculate_speed(fp_h_wall, overlap, drone.fps)

    walls = [
        ('y', wall_offset),
        ('y', l - wall_offset),
        ('x', wall_offset),
        ('x', w - wall_offset)
    ]
    for axis, pos in walls:
        span = w if axis == 'y' else l
        height_range = h - 2*clearance
        raw_wps = generate_wall_waypoints(
            span - 2*wall_offset,
            height_range, ss, sh,
            pos, axis, 0.0, speed_z
        )
        for wp in raw_wps:
            if axis == 'y':
                wp.x += wall_offset
            else:
                wp.y += wall_offset
            wp.z = max(wp.z + clearance, drone.min_altitude)
        all_wps.extend(raw_wps)

    return all_wps


def export_to_marvelmind(waypoints: List[Waypoint], filename: str):
    """
    write waypoints to csv columns: index, x, y, z, speed, gimbal_pitch
    each waypoint is numbered sequentially starting from 1
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'x', 'y', 'z', 'speed', 'gimbal_pitch'])
        for idx, wp in enumerate(waypoints, start=1):
            writer.writerow([idx, wp.x, wp.y, wp.z, wp.speed, wp.gimbal_pitch])


def visualize_waypoints_2d(
    waypoints: List[Waypoint],
    draw_lines: bool = True,
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None
):
    """
    plot top-down, side, and front projections with optional connecting lines

    draw_lines: if true, connect points in sequence
    line_kwargs: kwargs passed to plot() for lines
    marker_kwargs: kwargs passed to scatter() for markers
    """
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    line_kwargs = line_kwargs or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.7, 'color': 'C0'}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.8, 'color': 'C1'}

    fig, axs = plt.subplots(1, 3, figsize=(18,6))
    # top-down
    ax = axs[0]
    if draw_lines:
        ax.plot(xs, ys, **line_kwargs)
    ax.scatter(xs, ys, **marker_kwargs)
    ax.set(title='top-down', xlabel='x', ylabel='y')
    ax.grid(True)
    ax.axis('equal')

    # side
    ax = axs[1]
    if draw_lines:
        ax.plot(xs, zs, **line_kwargs)
    ax.scatter(xs, zs, **marker_kwargs)
    ax.set(title='side', xlabel='x', ylabel='z')
    ax.grid(True)

    # front
    ax = axs[2]
    if draw_lines:
        ax.plot(ys, zs, **line_kwargs)
    ax.scatter(ys, zs, **marker_kwargs)
    ax.set(title='front', xlabel='y', ylabel='z')
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_waypoints_3d(
    waypoints: List[Waypoint],
    elev: float = 30,
    azim: float = 45,
    draw_lines: bool = True,
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None
):
    """
    3d scatter plot with optional connecting lines

    draw_lines: if true, connect points in sequence
    line_kwargs: kwargs for plot()
    marker_kwargs: kwargs for scatter()
    """
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    line_kwargs = line_kwargs or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.8, 'color': 'C0'}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.9, 'color': 'C2'}

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    if draw_lines:
        ax.plot(xs, ys, zs, **line_kwargs)
    ax.scatter(xs, ys, zs, **marker_kwargs)
    ax.set(title='3d view', xlabel='x', ylabel='y', zlabel='z')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def validate_mission(drone: DroneConfig, waypoints: List[Waypoint]) -> Tuple[bool, dict]:
    """
    compute total flight distance and time, compare against drone limits

    returns (feasible, {'distance':..., 'time':...})
    """
    total_dist = 0.0
    total_time = 0.0
    for p1, p2 in zip(waypoints, waypoints[1:]):
        dx, dy, dz = p2.x-p1.x, p2.y-p1.y, p2.z-p1.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        total_dist += dist
        spd = p1.speed if p1.speed>0 else drone.speed
        total_time += dist/spd
    last = waypoints[-1]
    rt_dist = math.sqrt(last.x**2 + last.y**2 + last.z**2)
    total_dist += rt_dist
    total_time += rt_dist/(last.speed if last.speed>0 else drone.speed)

    allowed = min(drone.battery_time, drone.max_flight_time)
    return (total_dist<=drone.max_distance and total_time<=allowed,
            {'distance': total_dist, 'time': total_time})

if __name__ == '__main__':
    # example usage

    cfg = DroneConfig(
        battery_time=1200,          # max flight time (s)
        max_distance=5000,          # max travel distance (m)
        max_flight_time=1100,       # cap on mission duration (s)
        horizontal_fov=82.1,        # camera horizontal FOV (deg)
        vertical_fov=52.3,          # camera vertical FOV (deg)
        fps=60,                     # camera frame rate (fps)
        resolution=(2720, 1530),    # sensor resolution (px)
        speed=0.5,                  # nominal flight speed (m/s)
        min_altitude=1.0            # minimum safe altitude (m)
    )

    dims = (6.0, 6.0, 3.0)

    wps = generate_cube_scan(
        drone=cfg,
        dims=dims,
        overlap=0.7,                # 70% image overlap
        wall_offset=1.0,            # distance from walls (m)
        clearance=0.75               # inward margin from surfaces (m)
    )

    feasible, metrics = validate_mission(cfg, wps)

    scans_floor = sum(1 for wp in wps if wp.gimbal_pitch == -90.0)
    scans_ceiling = sum(1 for wp in wps if wp.gimbal_pitch == 60.0)
    scans_walls = len(wps) - scans_floor - scans_ceiling
    total_scans = len(wps)

    flight_time_s = metrics['time']
    flight_time_min = flight_time_s / 60
    battery_pct_used = flight_time_s / cfg.battery_time * 100

    print(f"scans per face: floor={scans_floor}, ceiling={scans_ceiling}, walls={scans_walls}")
    print(f"total scans: {total_scans}")
    print(f"total distance: {metrics['distance']:.1f} m")
    print(f"estimated flight time: {flight_time_s:.1f} s ({flight_time_min:.1f} min)")
    print(f"estimated battery usage: {flight_time_s:.1f} s ({battery_pct_used:.1f}% of {cfg.battery_time}s)")
    print(f"mission feasible: {feasible}")

    export_to_marvelmind(wps, 'waypoints.csv')
    visualize_waypoints_2d(wps)
    visualize_waypoints_3d(wps)
