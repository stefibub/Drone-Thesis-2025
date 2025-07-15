import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class WaypointPlanner:

    """
    Generate waypoints for full coverage of a cuboid space and export them as
    Marvelmind-compatible command scripts, and visualize in 3D.
    """
    def __init__(self, length, width, height, altitude, fov_deg,
                 sensor_ratio=(16,9), overlap=0.2, max_flight_time=2400,
                 wall_offset=None):
        self.length = length
        self.width = width
        self.height = height
        self.altitude = altitude
        self.fov = math.radians(fov_deg)
        self.sensor_ratio = sensor_ratio
        self.overlap = overlap
        self.max_flight_time = max_flight_time
        self.wall_offset = wall_offset or altitude

    def compute_ado(self):
        # area diagonal of coverage at altitude
        diag = 2 * self.altitude * math.tan(self.fov / 2)
        w_ratio, h_ratio = self.sensor_ratio
        r = math.hypot(w_ratio, h_ratio)
        return diag * w_ratio / r, diag * h_ratio / r

    def compute_grid(self):
        ado_w, ado_h = self.compute_ado()
        return ado_w * (1 - self.overlap), ado_h * (1 - self.overlap)

    def boustrophedon(self, nx, ny, dx, dy, origin, fixed_z, yaw, pitch):
        ox, oy = origin
        pts = []
        for j in range(ny):
            cols = range(nx) if j % 2 == 0 else range(nx-1, -1, -1)
            for i in cols:
                x = ox + i * dx
                y = oy + j * dy
                pts.append({'x': x, 'y': y, 'z': fixed_z, 'yaw': yaw, 'pitch': pitch})
        return pts

    def plan_floor(self):
        gw, gh = self.compute_grid()
        nx = math.ceil(self.length / gw)
        ny = math.ceil(self.width / gh)
        return self.boustrophedon(
            nx, ny,
            self.length / nx, self.width / ny,
            (0, 0), self.altitude,
            yaw=0, pitch=-90
        )

    def plan_wall(self, face):
        gw, gh = self.compute_grid()
        if face in ('north', 'south'):
            nx = math.ceil(self.length / gw)
            nz = math.ceil(self.height / gh)
            dx, dz = self.length / nx, self.height / nz
            y0 = self.width - self.wall_offset if face == 'north' else self.wall_offset
            yaw = 180 if face == 'north' else 0
            return self.boustrophedon(
                nx, nz,
                dx, dz,
                (0, y0), self.wall_offset,
                yaw, pitch=0
            )
        if face in ('east', 'west'):
            ny = math.ceil(self.width / gw)
            nz = math.ceil(self.height / gh)
            dy, dz = self.width / ny, self.height / nz
            x0 = self.length - self.wall_offset if face == 'east' else self.wall_offset
            yaw = -90 if face == 'east' else 90
            pts = []
            for j in range(nz):
                rows = range(ny) if j % 2 == 0 else range(ny-1, -1, -1)
                for i in rows:
                    x = x0
                    y = i * dy
                    z = self.wall_offset + j * dz
                    pts.append({'x': x, 'y': y, 'z': z, 'yaw': yaw, 'pitch': 0})
            return pts
        return []

    def plan_all(self):
        wps = []
        wps += self.plan_floor()
        for face in ['north', 'south', 'east', 'west']:
            wps += self.plan_wall(face)
        return wps

    def export_marvelmind(self, wps):
        """
        Create a Marvelmind DJI command script from waypoints:
        takeoff(), waypoints_begin(), Wnn(x,y,z,yaw), waypoints_end(), landing().
        """
        script = ['takeoff()', 'waypoints_begin()']
        for i, wp in enumerate(wps, 1):
            cmd = f"W{i:02d}({wp['x']:.2f},{wp['y']:.2f},{wp['z']:.2f},{int(wp['yaw'])})"
            script.append(cmd)
        script += ['waypoints_end()', 'landing()']
        return script

    def visualize(self, wps):
        """
        Plot waypoints in a 3D interactive grid. Works in PyCharm's SciView for rotation.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = [wp['x'] for wp in wps]
        ys = [wp['y'] for wp in wps]
        zs = [wp['z'] for wp in wps]
        ax.scatter(xs, ys, zs, marker='o')
        # connect in order
        ax.plot(xs, ys, zs)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_box_aspect((self.length, self.width, self.height))
        plt.title('Drone Waypoints')
        plt.show()

# example usage:
if __name__ == '__main__':
    planner = WaypointPlanner(10, 8, 3, 2.5, 75, (16,9), 0.2, 1800)
    waypoints = planner.plan_all()
    # visualize in 3D
    planner.visualize(waypoints)
    # export Marvelmind script
    commands = planner.export_marvelmind(waypoints)
    print('\n'.join(commands))