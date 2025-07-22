"""
optimized cube scan & planar coverage module

this module generates minimal waypoints for full cuboid coverage using optimized boustrophedon path planning
and accounts for turning radius, safe corridors, overlap, and camera fov constraints.
"""

import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class DroneConfig:
    battery_time: float
    max_distance: float
    max_flight_time: float
    horizontal_fov: float      # degrees
    vertical_fov: float        # degrees
    fps: float
    resolution: Tuple[int,int]
    speed: float               # nominal speed (m/s)
    min_altitude: float
    turning_radius: float      # meters
    hover_buffer: float        # seconds for stabilization

@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    gimbal_pitch: float
    speed: float
    hold_time: float           # seconds to hover


def calculate_footprint(drone: DroneConfig, distance: float) -> Tuple[float, float]:
    hfov = math.radians(drone.horizontal_fov)
    vfov = math.radians(drone.vertical_fov)
    w = 2 * distance * math.tan(hfov/2)
    h = 2 * distance * math.tan(vfov/2)
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
    """
    boustrophedon scan on planar surface with hover time:
      span_long: length of passes
      span_short: width between passes
      spacing: image spacing (fov*(1-overlap))
      hold_time: seconds to hover at each waypoint
    """
    n_passes = max(1, math.ceil(span_short / spacing))
    delta_short = span_short / n_passes
    waypoints: List[Waypoint] = []
    for i in range(n_passes + 1):
        coord_short = offset + i * delta_short
        if i % 2 == 0:
            start_long = offset + turning_radius
            end_long   = offset + span_long - turning_radius
        else:
            start_long = offset + span_long - turning_radius
            end_long   = offset + turning_radius
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
    speed_xy  = calculate_speed(fp_w, overlap, drone.fps)
    hold = 1.0 / drone.fps + drone.hover_buffer
    inset = wall_offset + clearance

    all_wps: List[Waypoint] = []

    # floor passes (x-aligned)
    all_wps += generate_planar_scan(
        span_long=w - 2*inset,
        span_short=l - 2*inset,
        spacing=spacing_x,
        z=max(clearance, drone.min_altitude),
        pitch=-90.0,
        speed=speed_xy,
        offset=inset,
        is_x_aligned=True,
        turning_radius=drone.turning_radius,
        hold_time=hold
    )

    # ceiling passes (y-aligned)
    all_wps += generate_planar_scan(
        span_long=l - 2*inset,
        span_short=w - 2*inset,
        spacing=spacing_y,
        z=max(h - clearance, drone.min_altitude),
        pitch=60.0,
        speed=speed_xy,
        offset=inset,
        is_x_aligned=False,
        turning_radius=drone.turning_radius,
        hold_time=hold
    )

    # walls
    fp_w_wall, fp_h_wall = calculate_footprint(drone, wall_offset)
    ss = fp_w_wall * (1 - overlap)
    sh = fp_h_wall * (1 - overlap)
    speed_z = calculate_speed(fp_h_wall, overlap, drone.fps)

    for axis, pos in [('y', wall_offset), ('y', l - wall_offset),
                      ('x', wall_offset), ('x', w - wall_offset)]:
        span = w if axis == 'y' else l
        height_range = h - 2*clearance
        nspan = max(1, math.ceil((span - 2*wall_offset) / ss))
        nheight = max(1, math.ceil(height_range / sh))
        ds = (span - 2*wall_offset) / nspan
        dh = height_range / nheight
        for i in range(nspan + 1):
            p_span = i * ds + wall_offset
            rows = (range(nheight + 1) if i % 2 == 0
                    else range(nheight, -1, -1))
            for j in rows:
                z_pt = j * dh + clearance
                x = p_span if axis == 'y' else pos
                y = pos    if axis == 'y' else p_span
                all_wps.append(
                    Waypoint(x, y, max(z_pt, drone.min_altitude),
                             0.0, speed_z, hold)
                )

    return all_wps

def export_to_marvelmind(waypoints: List[Waypoint], filename: str):
    # write index,x,y,z,HoldTime,GimblePitch to csv (CAN BE ADJUSTED ONCE IMPORT OF WAYPOINTS IS UNDERSTOOD)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'x', 'y', 'z', 'HoldTime', 'GimblePitch'])
        for idx, wp in enumerate(waypoints, start=1):
            writer.writerow([idx, wp.x, wp.y, wp.z, wp.hold_time, wp.gimbal_pitch])

def visualize_waypoints_2d(
    waypoints: List[Waypoint],
    draw_lines: bool = False,
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None
):
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    line_kwargs = line_kwargs or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.7, 'color': 'C2'}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.8, 'color': 'C1'}

    fig, axs = plt.subplots(1, 3, figsize=(18,6))
    for ax, (X, Y, title) in zip(
        axs,
        [(xs, ys, 'top-down'), (xs, zs, 'side'), (ys, zs, 'front')]
    ):
        if draw_lines:
            ax.plot(X, Y, **line_kwargs)
        ax.scatter(X, Y, **marker_kwargs)
        ax.set(title=title, xlabel='x' if title!='front' else 'y', ylabel='y' if title=='top-down' else 'z')
        ax.grid(True)
        if title=='top-down': ax.axis('equal')

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
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    line_kwargs = line_kwargs or {'linestyle': '-', 'linewidth': 1, 'alpha': 0.8, 'color': 'C2'}
    marker_kwargs = marker_kwargs or {'marker': 'o', 's': 30, 'alpha': 0.9, 'color': 'C1'}

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
    # compute travel time + hover time and compare to limits
    total_dist = 0.0
    travel_time = 0.0
    for p1, p2 in zip(waypoints, waypoints[1:]):
        dx, dy, dz = p2.x - p1.x, p2.y - p1.y, p2.z - p1.z
        dist = math.hypot(dx, dy, dz)
        total_dist += dist
        spd = p1.speed or drone.speed
        travel_time += dist / spd
    # return to origin
    last = waypoints[-1]
    rt = math.hypot(last.x, last.y, last.z)
    total_dist += rt
    travel_time += rt / (last.speed or drone.speed)
    # sum hold times
    hover_time = sum(wp.hold_time for wp in waypoints)
    total_time = travel_time + hover_time
    allowed = min(drone.battery_time, drone.max_flight_time)
    return (total_dist <= drone.max_distance and total_time <= allowed,
            {'distance': total_dist, 'time': total_time})

if __name__ == '__main__':
    # Example usage with new hover_buffer and turning_radius
    cfg = DroneConfig(
        battery_time=1200.0,        # max flight time (s)
        max_distance=5000.0,        # max travel distance (m)
        max_flight_time=1100.0,     # cap on mission duration (s)
        horizontal_fov=82.1,        # camera horizontal FOV (deg)
        vertical_fov=52.3,          # camera vertical FOV (deg)
        fps=60.0,                   # camera frame rate (fps)
        resolution=(2720, 1530),    # sensor resolution (px)
        speed=0.5,                  # nominal flight speed (m/s)
        min_altitude=1.0,           # minimum safe altitude (m)
        turning_radius=1.0,         # minimum turn radius (m)
        hover_buffer=1            # stabilization buffer (s)
    )

    # survey area dimensions: width, length, height (m)
    dims = (6.0, 6.0, 3.0)

    # generate optimized cube scan waypoints (hold_time set internally)
    wps = generate_cube_scan(
        drone=cfg,
        dims=dims,
        overlap=0.7,               # 70% image overlap
        wall_offset=2.0,           # horizontal offset from walls (m)
        clearance=0.75             # inward margin from surfaces (m)
    )

    # validate mission feasibility (includes hover time)
    feasible, metrics = validate_mission(cfg, wps)

    # count scans by face
    scans_floor   = sum(1 for wp in wps if wp.gimbal_pitch == -90.0)
    scans_ceiling = sum(1 for wp in wps if wp.gimbal_pitch ==  60.0)
    scans_walls   = len(wps) - scans_floor - scans_ceiling
    total_scans   = len(wps)

    # extract metrics
    total_distance = metrics['distance']
    total_time     = metrics['time']
    battery_pct    = (total_time / cfg.battery_time) * 100

    # output summary
    print(f"scans per face: floor={scans_floor}, ceiling={scans_ceiling}, walls={scans_walls}")
    print(f"total scans: {total_scans}")
    print(f"total distance: {total_distance:.1f} m")
    print(f"total flight time (including hover): {total_time:.1f} s ({total_time/60:.1f} min)")
    print(f"battery usage: {battery_pct:.1f}% of capacity")
    print(f"mission feasible: {feasible}")

    # export waypoints (with HoldTime) and visualize
    export_to_marvelmind(wps, 'waypoints.csv')
    visualize_waypoints_2d(wps)
    visualize_waypoints_3d(wps)
