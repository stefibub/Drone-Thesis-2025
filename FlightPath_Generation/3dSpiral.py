import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import os 
from datetime import datetime

@dataclass
class DroneConfig:
    """
    holds drone settings and limits
    """
    weight: float                    # weight in kg 
    max_battery_time: float          # max flight time in seconds - tested 
    max_distance: Optional[float]    # max travel distance in meters
    horizontal_fov: float            # camera horizontal field of view in degrees
    vertical_fov: float              # camera vertical field of view in degrees (will be recomputed)
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


###### Helical trajectory generation ######


def generate_helical_trajectory(
    radius: float,
    vertical_pitch: float,
    revolutions: float,
    points_per_revolution: int,
    start_z: float,
    gimbal_pitch: float,
    speed: float,
    hold_time: float
) -> List[Waypoint]:
    """
    3D helix path generator 
    - radius: horizontal radius of coil
    - vertical_pitch: vertical climb per full revolution
    - revolutions: how many turns 
    - points_per_revolution: resolution around each loop
    - start_z: starting altitude
    - gimbal_pitch: camera angle
    - speed: assignment for waypoint speed
    - hold_time: hover time at each waypoint
    """
    waypoints: List[Waypoint] = []
    total_points = max(1, int(revolutions * points_per_revolution))
    for i in range(total_points + 1):
        # it is the angle around circle. Each full revolution increments theta by 2π
        theta = 2 * math.pi * (i / points_per_revolution)
        room_center_x = room_width / 2.0
        room_center_y = room_length / 2.0
        x = room_center_x + radius * math.cos(theta)
        y = room_center_y + radius * math.sin(theta)
        # linear climb proportional to angle: each 2π increases z by vertical_pitch
        z = start_z + (vertical_pitch * (theta / (2 * math.pi)))
        waypoints.append(Waypoint(x, y, z, gimbal_pitch, speed, hold_time))
    return waypoints

##### Sampling and speed logic based on overlap and camera geometry #####
def compute_helix_sampling_for_forward_camera(
    radius: float,
    representative_range: float,
    hfov_deg: float,
    vfov_deg: float,
    overlap: float
) -> Tuple[int, float]:
    """
    how dense the helix should be:
    - points_per_revolution: how many samples around each loop to achieve desired lateral overlap
    - vertical_pitch: how much height is gained per loop to achieve vertical overlap

    using a forward-facing camera model (assuming gimble is not moved):
      - representative_range: distance ahead the camera is imaging (e.g., nearest wall)
      - hfov_deg, vfov_deg: field of view angles
      - overlap: desired fractional overlap between adjacent views (e.g., 0.7)
    """
    if not (0 < overlap < 1):
        raise ValueError("overlap must be between 0 and 1 (exclusive)")
    s = 1 - overlap
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)
    
    # approximate size of what the camera sees at that forward distance -> tbd
    footprint_width = 2 * representative_range * math.tan(hfov / 2)
    footprint_height = 2 * representative_range * math.tan(vfov / 2)

    # lateral allowed movement per loop step to keep overlap
    lateral_step = footprint_width * s
    if radius <= 0 or lateral_step <= 0:
        raise ValueError("Invalid radius or overlap setting.")

    # converting lateral step into angular resolution around circle
    delta_theta = lateral_step / radius
    if delta_theta <= 0:
        delta_theta = 1e-6 # safety reasons 
    points_per_revolution = math.ceil(2 * math.pi / delta_theta)

    # vertical pitch to ensure adjacent coils overlap vertically
    vertical_pitch = footprint_height * s

    return points_per_revolution, vertical_pitch


def compute_helix_speed_for_overlap(
    radius: float,
    vertical_pitch: float,
    hfov_deg: float,
    vfov_deg: float,
    representative_range: float,
    overlap: float,
    fps: float
) -> float:
    """
    estimation of helix traversal speed such that frame-to-frame movement at `fps`
    stays within the overlap bounds, considering both tangential (around circle)
    and vertical components.
    """
    if not (0 < overlap < 1):
        raise ValueError("overlap must be between 0 and 1 (exclusive)")
    s = 1 - overlap
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)

    # footprint size at representative forward distance
    footprint_width = 2 * representative_range * math.tan(hfov / 2)
    footprint_height = 2 * representative_range * math.tan(vfov / 2)

    # allowed per-frame movements
    max_lateral_per_frame = footprint_width * s
    max_vertical_per_frame = footprint_height * s

    # in angular terms, how much the camera can move per frame
    delta_theta_lateral = max_lateral_per_frame / radius if radius > 0 else float('inf')
    delta_theta_vertical = (2 * math.pi * max_vertical_per_frame) / vertical_pitch if vertical_pitch > 0 else float('inf')

    # stricter constraint (smallest angle)
    delta_theta_frame = min(delta_theta_lateral, delta_theta_vertical)
    omega = delta_theta_frame * fps # angular speed in rad/s

    v_circ = radius * omega
    v_vert = (vertical_pitch / (2 * math.pi)) * omega
    total_speed = math.sqrt(v_circ**2 + v_vert**2)
    return total_speed


###### Processing ######

def visualize_waypoints_2d(
    waypoints: List[Waypoint],
    draw_lines: bool = False,
    line_kwargs: Optional[Dict] = None,
    marker_kwargs: Optional[Dict] = None
):
    """
    2D projections: top-down (x,y), side (x,z), front (y,z)
    """
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
    3D view of the helix path.
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
    Mission feasibility: distance, time, and battery usage
    Always returns to origin at end
    """
    total_distance = 0.0
    total_travel_time = 0.0

    if not waypoints:
        return False, {}

    for wp1, wp2 in zip(waypoints, waypoints[1:]):
        dx, dy, dz = wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        total_distance += dist
        total_travel_time += dist / drone.speed

    last_wp = waypoints[-1]
    return_dist = math.sqrt(last_wp.x**2 + last_wp.y**2 + last_wp.z**2)
    return_time = return_dist / drone.speed
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
    # basic drone configuration
    cfg = DroneConfig(
        weight         = 0.292,
        max_battery_time   = 1380.0,
        max_distance   = None,
        horizontal_fov = 82.1,
        vertical_fov   = 52.3,  # placeholder; will recompute from aspect
        fps            = 60.0,
        resolution     = (2720, 1530),
        speed          = 0.5,
        min_altitude   = 1.0,
        turning_radius = 1.0,
        hover_buffer   = 2,
        battery_warning_threshold= 0.85
    )

    # room definition and margins
    dims = (6.0, 6.0, 3.0)  # width, length, height in meters
    clearance = 0.2         # vertical inset from ceiling
    margin = 0.5            # horizontal inset from walls
    start_z = 0.5           # starting altitude

    room_width, room_length, room_height = dims
    radius = max(0.0, min(room_width, room_length) / 2.0 - margin)
    max_height = room_height - clearance # top of helix

    # recompute vertical fov from aspect ratio to keep projection consistent
    aspect = cfg.resolution[1] / cfg.resolution[0]  # height / width
    hfov_rad = math.radians(cfg.horizontal_fov)
    vfov_rad = 2 * math.atan(math.tan(hfov_rad / 2) * aspect)
    vertical_fov_deg = math.degrees(vfov_rad)

    # desired video overlap and range
    overlap = 0.7  # 70%
    representative_range = radius  # nearest wall ahead assumption

    # compute helix density from overlap
    points_per_rev, vertical_pitch = compute_helix_sampling_for_forward_camera(
        radius=radius,
        representative_range=representative_range,
        hfov_deg=cfg.horizontal_fov,
        vfov_deg=vertical_fov_deg,
        overlap=overlap
    )

    # compute how many (full/partial) revolutions fit in vertical space
    available_climb = max_height - start_z
    revolutions = available_climb / vertical_pitch if vertical_pitch > 0 else 0.0

    # compute target speed to maintain overlap (capped by nominal)
    target_speed = compute_helix_speed_for_overlap(
        radius=radius,
        vertical_pitch=vertical_pitch,
        hfov_deg=cfg.horizontal_fov,
        vfov_deg=vertical_fov_deg,
        representative_range=representative_range,
        overlap=overlap,
        fps=cfg.fps
    )
    helix_speed = min(target_speed, cfg.speed)

    # debugging
    footprint_width = 2 * representative_range * math.tan(hfov_rad / 2)
    footprint_height = 2 * representative_range * math.tan(vfov_rad / 2)
    print("=== Helix sampling diagnostics ===\n")
    print(f"Footprint (WxH) at D=radius: {footprint_width:.2f} x {footprint_height:.2f} m")
    print(f"Overlap target: {overlap*100:.0f}%")
    print(f"Points per revolution: {points_per_rev}")
    print(f"Vertical pitch (per revolution): {vertical_pitch:.2f} m")
    print(f"Revolutions fitting in climb ({available_climb:.2f} m): {revolutions:.2f}")
    print(f"Estimated total waypoints: {points_per_rev * revolutions:.1f}")
    print(f"Helix speed used: {helix_speed:.2f} m/s")

    # generation of the helical trajectory
    helix_wps = generate_helical_trajectory(
        radius=radius,
        vertical_pitch=vertical_pitch,
        revolutions=revolutions,
        points_per_revolution=points_per_rev,
        start_z=start_z,
        gimbal_pitch=-45.0,
        speed=helix_speed,
        hold_time=0.5
    )

    feasible, metrics = validate_mission(cfg, helix_wps)

    out_dir = "3DSpiral_FlightPlanData"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"mission_path_lawnmower_2d_{timestamp}"

    # output results 
    print("\n=== Helical trajectory generated ===")
    print(f"\nGenerated helix with {len(helix_wps)} waypoints.")
    print(f"Derived parameters: radius={radius:.2f}, vertical_pitch={vertical_pitch:.2f}, "
          f"revolutions={revolutions:.2f}, points_per_rev={points_per_rev}, helix_speed={helix_speed:.2f} m/s")
    print(f"Total distance (including return to origin): {metrics['distance']:.1f} m")
    print(f"Total mission time: {metrics['total_time']:.1f} s ({metrics['total_time']/60:.1f} min)")
    print(f"Battery usage: {metrics['battery_usage_percent']:.1f}%")
    print(f"Mission feasible: {feasible}")
    if metrics['battery_warning']:
        print("Warning: Battery usage exceeds safe threshold.")

        # consistent filenames
    csv_filename = os.path.join(out_dir, f"{base_name}.csv")
    txt_filename = os.path.join(out_dir, f"{base_name}.txt")
    dpt_filename = os.path.join(out_dir, f"{base_name}.dpt")

    # Export
    export_to_marvelmind(helix_wps, csv_filename)
    export_to_marvelmind(helix_wps, txt_filename)
    export_to_dpt(helix_wps, cfg, dpt_filename)

    print(f"Export Complete:\n{csv_filename},\n{txt_filename},\n{dpt_filename}.\n")
    visualize_waypoints_2d(helix_wps, draw_lines=True)
    visualize_waypoints_3d(helix_wps, draw_lines=True)

