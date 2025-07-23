import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt

@dataclass
class DroneConfig:
    """
    holds drone settings and limits
    """
    weight: float                    # weight in kg 
    max_battery_time: float          # max flight time in seconds - tested 
    max_distance: Optional[float]              # max travel distance in meters
    horizontal_fov: float            # camera horizontal field of view in degrees
    vertical_fov: float              # camera vertical field of view in degrees
    fps: float                       # camera frame rate in frames per second
    resolution: Tuple[int,int]       # camera resolution in pixels (width, height)
    speed: float                     # nominal flight speed in m/s
    min_altitude: float              # minimum safe flight altitude in meters
    turning_radius: float            # minimum turn radius in meters
    hover_buffer: float              # extra hover time for stabilization in seconds
    battery_warning_threshold: float # warning if battery % used exceeds this

    def __post_init__(self):
        if self.max_distance is None:
            self.max_distance = self.speed * self.max_battery_time

@dataclass
class Waypoint:
    """
    stores a single waypoint for the mission
    """
    x: float                     # x-coordinate in meters
    y: float                     # y-coordinate in meters
    z: float                     # z-coordinate (altitude) in meters
    gimbal_pitch: float          # camera pitch angle in degrees
    speed: float                 # travel speed to next waypoint in m/s
    hold_time: float             # hover time at this point in seconds


def calculate_footprint(drone: DroneConfig, distance: float) -> Tuple[float, float]:
    """
    compute footprint size at given distance
    returns (width, height) in meters based on camera fov using pinhole equations
    """
    # convert degrees to radians
    hfov = math.radians(drone.horizontal_fov)
    vfov = math.radians(drone.vertical_fov)
    # compute width and height of view at given distance
    w = 2 * distance * math.tan(hfov/2)
    h = 2 * distance * math.tan(vfov/2)
    return w, h


def calculate_speed(footprint_len: float, overlap: float, fps: float) -> float:
    """
    compute speed to maintain desired overlap
    footprint_len: size of one image in meters
    overlap: fraction between 0 and 1
    fps: camera frame rate
    """
    # effective distance between frames
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
    generate boustrophedon waypoints on a flat surface
    span_long: length of scan area
    span_short: width of scan area
    spacing: distance between images
    z: altitude
    pitch: camera pitch angle
    speed: travel speed
    offset: inset from edges
    is_x_aligned: scan direction flag
    turning_radius: margin to avoid sharp turns
    hold_time: hover time per point
    """
    # compute number of passes needed
    n_passes = max(1, math.ceil(span_short / spacing))
    # actual gap between passes
    delta_short = span_short / n_passes
    waypoints: List[Waypoint] = []
    # loop over each pass
    for i in range(n_passes + 1):
        coord_short = offset + i * delta_short
        # decide start and end of this pass to account for turning radius
        if i % 2 == 0:
            start_long = offset + turning_radius
            end_long   = offset + span_long - turning_radius
        else:
            start_long = offset + span_long - turning_radius
            end_long   = offset + turning_radius
        # length of this pass
        seg_len = abs(end_long - start_long)
        # number of samples (images) along pass
        n_samples = max(1, math.ceil(seg_len / spacing))
        delta_long = (end_long - start_long) / n_samples
        # generate points along this pass
        for j in range(n_samples + 1):
            pos_long = start_long + j * delta_long
            # assign x,y based on alignment
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
    """
    create waypoints to cover all faces of a cuboid
    dims: (width, length, height) in meters
    overlap: image overlap fraction
    wall_offset: horizontal gap from walls
    clearance: inset from surfaces
    """
    w, l, h = dims
    # footprint size at ceiling height
    fp_w, fp_h = calculate_footprint(drone, h)
    # spacing in x and y
    spacing_x = fp_w * (1 - overlap)
    spacing_y = fp_h * (1 - overlap)
    # speed in horizontal plane
    speed_xy  = calculate_speed(fp_w, overlap, drone.fps)
    # hover time per point (one frame + buffer)
    hold = 1.0 / drone.fps + drone.hover_buffer
    # distance from walls and floor/ceiling
    inset = wall_offset + clearance

    all_wps: List[Waypoint] = []

    # floor scan (camera downwards)
    all_wps += generate_planar_scan(
        span_long = w - 2*inset,
        span_short= l - 2*inset,
        spacing   = spacing_x,
        z         = max(clearance, drone.min_altitude),
        pitch     = -90.0,
        speed     = speed_xy,
        offset    = inset,
        is_x_aligned     = True,
        turning_radius   = drone.turning_radius,
        hold_time        = hold
    )

    # ceiling scan (angled camera)
    all_wps += generate_planar_scan(
        span_long = l - 2*inset,
        span_short= w - 2*inset,
        spacing   = spacing_y,
        z         = max(h - clearance, drone.min_altitude),
        pitch     = 60.0,
        speed     = speed_xy,
        offset    = inset,
        is_x_aligned     = False,
        turning_radius   = drone.turning_radius,
        hold_time        = hold
    )

    # wall scans
    # compute footprint at wall_offset
    fp_w_wall, fp_h_wall = calculate_footprint(drone, wall_offset)
    ss = fp_w_wall * (1 - overlap)
    sh = fp_h_wall * (1 - overlap)
    speed_z = calculate_speed(fp_h_wall, overlap, drone.fps)

    # loop over each wall side
    for axis, pos in [('y', wall_offset), ('y', l - wall_offset),
                      ('x', wall_offset), ('x', w - wall_offset)]:
        # length along wall and height range
        span = w if axis == 'y' else l
        height_range = h - 2*clearance
        # number of horizontal and vertical passes
        nspan = max(1, math.ceil((span - 2*wall_offset) / ss))
        nheight = max(1, math.ceil(height_range / sh))
        ds = (span - 2*wall_offset) / nspan
        dh = height_range / nheight
        # generate grid on wall
        for i in range(nspan + 1):
            p_span = i * ds + wall_offset
            # zigzag row order for smooth path
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
    """
    export waypoints to csv file for marvelmind import
    columns: index, x, y, z, hold time, gimbal pitch
    """
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
    """
    plot top-down, side, and front views of waypoints in 2d
    draw_lines: connect points if true
    """
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    # default style settings
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
        ax.set(title=title,
               xlabel='x' if title!='front' else 'y',
               ylabel='y' if title=='top-down' else 'z')
        ax.grid(True)
        if title=='top-down':
            ax.axis('equal')

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
    plot 3d view of waypoints
    elev: elevation angle, azim: azimuth angle
    """
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
    """
    Evaluate mission feasibility using tested max battery time.
    Calculates travel time and hover time, then checks if the mission fits within limits.
    Returns feasibility as boolean and dictionary of metrics. 
    """
    total_distance = 0.0
    total_travel_time = 0.0

    for wp1, wp2 in zip(waypoints, waypoints[1:]):
        dx, dy, dz = wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        speed = drone.speed
        # speed = wp1.speed or drone.speed
        #print(f"Segment: ({wp1.x:.1f},{wp1.y:.1f},{wp1.z:.1f}) → ({wp2.x:.1f},{wp2.y:.1f},{wp2.z:.1f}), dist={dist:.2f}, speed={speed:.2f}, time={dist/speed:.2f}")
        total_distance += dist
        total_travel_time += dist / speed

    # return to origin
    last_wp = waypoints[-1]
    return_dist = math.sqrt(last_wp.x**2 + last_wp.y**2 + last_wp.z**2)
    return_time = return_dist / drone.speed #this could have same problem w1.speed -> needs to be changed to drone speed 
    print(f"The speed of the last waypoint is: {last_wp.speed}, the return distance is: {return_dist},the return time is {return_time}")
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



if __name__ == '__main__':
    # example usage with default settings
    cfg = DroneConfig(
        weight         = 0.292,          # kg
        max_battery_time   = 1380.0,     # seconds
        max_distance   = None,           # meters
        horizontal_fov = 82.1,           # degrees
        vertical_fov   = 52.3,           # degrees
        fps            = 60.0,           # frames per second
        resolution     = (2720, 1530),
        speed          = 0.5,            # m/s
        min_altitude   = 1.0,            # meters
        turning_radius = 1.0,            # meters
        hover_buffer   = 2,              # seconds
        battery_warning_threshold= 0.85  # percentage
    ) 

    dims = (6.0, 6.0, 3.0)         # width, length, height in meters
    # generate waypoints for full cube scan
    wps = generate_cube_scan(
        drone      = cfg,
        dims       = dims,
        overlap    = 0.7,           # 70% overlap
        wall_offset= 2.0,           # meters from walls
        clearance  = 0.75           # meters margin
    )

    # validate mission feasibility
    feasible, metrics = validate_mission(cfg, wps)
    scans_floor   = sum(1 for wp in wps if wp.gimbal_pitch == -90.0)
    scans_ceiling = sum(1 for wp in wps if wp.gimbal_pitch == 60.0)
    scans_walls   = len(wps) - scans_floor - scans_ceiling
    total_scans   = len(wps)
    total_distance = metrics['distance']

####################### print summary #######################

    #print(f"Travel time: {metrics['travel_time']:.1f} s")
    #print(f"Hover time: {metrics['hover_time']:.1f} s")
    print(f"scans per face: floor={scans_floor}, ceiling={scans_ceiling}, walls={scans_walls}")
    print(f"total scans: {total_scans}")
    print(f"total distance: {total_distance:.1f} m")
    print(f"Total mission time: {metrics['total_time']:.1f} s ({metrics['total_time']/60:.1f} min) "
          f"(includes travel and hover time)")
    print(f"Battery usage: {metrics['battery_usage_percent']:.1f}% of capacity")
    print(f"Mission feasible: {feasible}")

    if metrics['battery_warning']:
        print("Warning: Battery usage exceeds safe threshold "
              f"({cfg.battery_warning_threshold}%) — mission is near the limit.")


    # export and visualize
    export_to_marvelmind(wps, 'waypoints.csv')
    visualize_waypoints_2d(wps)
    visualize_waypoints_3d(wps)
