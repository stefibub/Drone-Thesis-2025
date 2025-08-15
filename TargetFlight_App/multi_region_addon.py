import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate


# ========================================================================================================================
# ====================================                  MULTI REGION ADDON                  ==============================
#
#  Extension module for the Target Flightpath Generator. Provides functionality to expand a single anchor voxel into a 
#  larger, custom-shaped region and generate a flight path tailored to that region.
#
#  Purpose:
#   - allow users to extend the default single-cube selection into multi-voxel regions
#   - calculate bounding box and geometric center of the expanded region
#   - generate a single orbital flight path around the region’s center
#   - provide feedback when user growth requests exceed grid limits (auto-clamps to maximum)
#   - visualize the expanded region with bounding box edges
#
#  Features:
#   - region growth in ±X, ±Y, ±Z directions with grid-aware clamping
#   - effective vs requested growth reporting (blocked directions flagged)
#   - single orbit generation reusing host’s cube orbit logic
#   - helper utilities for bounds, region IDs, and visualization edges
# ========================================================================================================================

#---types from host---
DroneConfig = None
VoxelGrid   = None
Waypoint    = None

# ---------- region helpers ----------
@dataclass
class RegionExtents:
    """hold per-axis growth (in cells) away from the anchor voxel."""
    x_minus: int = 0
    x_plus:  int = 0
    y_minus: int = 0
    y_plus:  int = 0
    z_minus: int = 0
    z_plus:  int = 0

def _clamp(v, lo, hi):
    """return v clamped to [lo, hi]."""
    return max(lo, min(hi, v))

def get_region_ids(grid, anchor_id: int, ext: RegionExtents) -> List[int]:
    """
    return a face-connected rectangular prism of ids grown from anchor_id by 'ext' cells
    along each axis. Stays inside the grid.
    """
    #anchor indices
    i0, j0, k0 = grid.id_to_ijk(anchor_id)
    #compute inclusive bounds in index space, clamped to grid limits
    i_min = _clamp(i0 - int(ext.x_minus), 0, grid.nx - 1)
    i_max = _clamp(i0 + int(ext.x_plus),  0, grid.nx - 1)
    j_min = _clamp(j0 - int(ext.y_minus), 0, grid.ny - 1)
    j_max = _clamp(j0 + int(ext.y_plus),  0, grid.ny - 1)
    k_min = _clamp(k0 - int(ext.z_minus), 0, grid.nz - 1)
    k_max = _clamp(k0 + int(ext.z_plus),  0, grid.nz - 1)

    #enumerate all voxels in the prism (face-connected block)
    ids = []
    for k in range(k_min, k_max + 1):
        for j in range(j_min, j_max + 1):
            for i in range(i_min, i_max + 1):
                ids.append(grid.ijk_to_id(i, j, k))
    return ids

def region_bounds(grid, ids: List[int]) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """axis-aligned bounds of a set of voxel ids (min corner, max corner)."""
    if not ids:
        return (0,0,0), (0,0,0)
    mins = [float("inf")]*3
    maxs = [float("-inf")]*3
    for id1 in ids:
        i,j,k = grid.id_to_ijk(id1)
        bmin, bmax = grid.index_to_bounds(i,j,k)
        for a in range(3):
            mins[a] = min(mins[a], bmin[a])
            maxs[a] = max(maxs[a], bmax[a])
    return tuple(mins), tuple(maxs)

def cube_edges_from_bounds(bounds_min, bounds_max):
    """return 3 arrays (x,y,z) describing wireframe edges for a box defined by bounds."""
    xmin,ymin,zmin = bounds_min
    xmax,ymax,zmax = bounds_max

    #corner list in a fixed order
    corners = np.array([
        [xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin],
        [xmin,ymin,zmax],[xmax,ymin,zmax],[xmax,ymax,zmax],[xmin,ymax,zmax],
    ])

    #edge indices across corners list
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a,b in edges:
        xs += [corners[a,0], corners[b,0], None]
        ys += [corners[a,1], corners[b,1], None]
        zs += [corners[a,2], corners[b,2], None]
    return xs, ys, zs


def _clamped_growth_feedback(grid, anchor_id: int, ext: RegionExtents):
    """
    Compare requested growth vs maximum possible from the anchor (within grid).
    Returns:
      eff  : dict of effective growth per direction (integers)
      blocked_dirs : list of direction labels that were partially/fully blocked by bounds
      any_requested : did the user request any positive growth?
      any_effective : did any growth actually take effect?
      req : dict of requested growth per direction
    """
    #anchor location in index space
    i0, j0, k0 = grid.id_to_ijk(anchor_id)

    #max growth available from anchor to boundaries
    max_minus = {'-X': i0, '-Y': j0, '-Z': k0}
    max_plus  = {'+X': grid.nx - 1 - i0, '+Y': grid.ny - 1 - j0, '+Z': grid.nz - 1 - k0}
    #requested growth per direction
    req = {
        '-X': int(ext.x_minus or 0), '+X': int(ext.x_plus or 0),
        '-Y': int(ext.y_minus or 0), '+Y': int(ext.y_plus or 0),
        '-Z': int(ext.z_minus or 0), '+Z': int(ext.z_plus or 0),
    }
    #compute effective growth after clamping
    eff = {}
    blocked = []

    #minus directions
    for d in ['-X','-Y','-Z']:
        allowed = max_minus[d]
        eff[d] = max(0, min(req[d], allowed))
        if req[d] > eff[d]:
            blocked.append(d)

    #plus directions
    for d in ['+X','+Y','+Z']:
        allowed = max_plus[d]
        eff[d] = max(0, min(req[d], allowed))
        if req[d] > eff[d]:
            blocked.append(d)

    any_requested = any(v > 0 for v in req.values())
    any_effective = any(v > 0 for v in eff.values())

    return eff, blocked, any_requested, any_effective, req



# ---------- planning helpers ----------
def region_center_from_ids(grid, ids: List[int]) -> Tuple[float, float, float]:
    """Center of the region’s axis-aligned bounding box."""
    if not ids:
        return (0.0, 0.0, 0.0)
    bmin, bmax = region_bounds(grid, ids)
    return tuple((bmin[a] + bmax[a]) * 0.5 for a in range(3))


def generate_region_orbits(
    grid, drone, ids: List[int],
    generate_cube_orbit_fn,
    radius: float, n_views: int, dwell_s: float,
    floor_clear: float, ceil_clear: float
) -> List:
    """build a single orbit at the region center using the host's generate_cube_orbit."""
    if not ids:
        return []
    center = region_center_from_ids(grid, ids)

    #one ring, same behavior as single-voxel orbit (pruning + possible shrink handled inside)
    wps = generate_cube_orbit_fn(
        drone=drone, dims=grid.dims, center=center,
        radius=radius, n_views=max(3, int(n_views)), dwell_s=dwell_s,
        wall_clearance=0.02, floor_clearance=floor_clear, ceiling_clearance=ceil_clear
    )
    return wps


# ---------- rendering ----------
def make_region_figure(grid, ids: List[int], waypoints, title="Region preview"):
    W,L,H = grid.dims
    fig = go.Figure()
    #room
    room_corners = np.array([
        [0,0,0],[W,0,0],[W,L,0],[0,L,0],
        [0,0,H],[W,0,H],[W,L,H],[0,L,H],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    rx,ry,rz = [],[],[]
    for a,b in edges:
        rx += [room_corners[a,0], room_corners[b,0], None]
        ry += [room_corners[a,1], room_corners[b,1], None]
        rz += [room_corners[a,2], room_corners[b,2], None]
    fig.add_trace(go.Scatter3d(x=rx,y=ry,z=rz,mode='lines',line=dict(width=2),name="Room"))

    #region box
    if ids:
        bmin, bmax = region_bounds(grid, ids)
        ex,ey,ez = cube_edges_from_bounds(bmin,bmax)
        fig.add_trace(go.Scatter3d(x=ex,y=ey,z=ez,mode='lines',line=dict(width=6),name="Region"))
        xmin,ymin,zmin = bmin; xmax,ymax,zmax = bmax
        corners = np.array([
            [xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin],
            [xmin,ymin,zmax],[xmax,ymin,zmax],[xmax,ymax,zmax],[xmin,ymax,zmax]
        ])
        i_surf=[0,1,2,3,4,5,6,7,0,1,5,4]
        j_surf=[1,2,3,0,5,6,7,4,4,5,6,7]
        k_surf=[4,5,6,7,0,1,2,3,1,2,6,5]
        fig.add_trace(go.Mesh3d(
            x=corners[:,0], y=corners[:,1], z=corners[:,2],
            i=i_surf, j=j_surf, k=k_surf, opacity=0.12, color='lightgreen', name="Region volume"
        ))

    #path
    if waypoints:
        xs=[wp.x for wp in waypoints]; ys=[wp.y for wp in waypoints]; zs=[wp.z for wp in waypoints]
        fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode='lines+markers',marker=dict(size=3),name="Flight path"))

    fig.update_layout(
        title=title, height=650, margin=dict(l=0,r=0,t=30,b=0),
        scene=dict(
            xaxis=dict(range=[0,W],title="X (m)"),
            yaxis=dict(range=[0,L],title="Y (m)"),
            zaxis=dict(range=[0,H],title="Z (m)"),
            aspectmode='data'
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig

# ---------- exposed to host ----------
def multi_hidden_components():
    """invisible stores the host app should push in its layout."""
    return [
        dcc.Store(id="multi-flight-summary-json"),
        dcc.Store(id="multi-export-payloads"),
    ]

# ---------- add-on UI ----------
def render_panel():
    """return the multi-region configuration panel."""
    return html.Div(
        [
            html.H3("Multi-cube region", style={"margin":"0 0 8px 0","fontWeight":"600"}),
            html.Div("Extend the currently selected cube into a face-connected rectangle.",
                     style={"color":"#555","fontSize":"12px","marginBottom":"6px"}),
            #growth inputs per axis/side
            html.Div([
                html.Div([html.Label("−X"), dcc.Input(id="mr-xm", type="number", value=0, min=0, step=1)]),
                html.Div([html.Label("+X"), dcc.Input(id="mr-xp", type="number", value=0, min=0, step=1)]),
                html.Div([html.Label("−Y"), dcc.Input(id="mr-ym", type="number", value=0, min=0, step=1)]),
                html.Div([html.Label("+Y"), dcc.Input(id="mr-yp", type="number", value=0, min=0, step=1)]),
                html.Div([html.Label("−Z"), dcc.Input(id="mr-zm", type="number", value=0, min=0, step=1)]),
                html.Div([html.Label("+Z"), dcc.Input(id="mr-zp", type="number", value=0, min=0, step=1)]),
            ], style={"display":"grid","gridTemplateColumns":"repeat(6, minmax(70px,1fr))","gap":"8px"}),

            html.Br(),
            #orbit params
            html.Label("Orbit params:", style={"fontSize":"12px","color":"#555"}),
            html.Div([
                html.Div([html.Label("Orbit radius (m)"), dcc.Input(id="mr-radius", type="number", value=0.6, min=0.2, step=0.1)]),
                html.Div([html.Label("Views (per orbit)"), dcc.Input(id="mr-views", type="number", value=8, min=3, step=1)]),
                html.Div([html.Label("Hover time per view (s)"),  dcc.Input(id="mr-dwell",  type="number", value=1.5, min=0.0, step=0.5)]),
            ], style={"display":"grid","gridTemplateColumns":"repeat(3, minmax(120px,1fr))","gap":"8px"}),

            html.Br(),
            html.Div(
                [
                    html.Button(
                        "Preview region",
                        id="mr-preview",
                        n_clicks=0,
                        style={
                            "padding": "8px 12px",
                            "borderRadius": "8px",
                            "background": "#2563eb",
                            "color": "white",
                            "border": "1px solid #1d4ed8",
                        },
                    ),
                ],
                style={
                    "display":"flex",
                    "alignItems":"center",
                    "justifyContent":"center",
                    "flexWrap":"wrap",
                },
            ),

            html.Div(id="mr-status", style={"marginTop":"8px","fontFamily":"monospace","fontSize":"12px","color":"#374151"}),

            html.Div([dcc.Graph(id="mr-graph")],
                     style={"border":"1px solid #eaeaea","borderRadius":"12px","padding":"8px","marginTop":"10px","background":"white"}),
        ],
        style={"border":"1px solid #eaeaea","borderRadius":"12px","padding":"12px","background":"white","boxShadow":"0 1px 2px rgba(0,0,0,0.04)","marginTop":"12px"}
    )

# ---------- mounting ----------
def mount_multi_region(app, grid_ref, drone_ref, generate_cube_orbit_fn, waypoint_cls, voxelgrid_cls):
    """
    attach the multi-region panel and callbacks to the host app.
    """
    global DroneConfig, VoxelGrid, Waypoint
    DroneConfig = type(drone_ref)
    VoxelGrid   = voxelgrid_cls
    Waypoint    = waypoint_cls

    # --- hidden stores for host ---
    panel_wrapper = html.Div(
        id="mr-panel-wrapper",
        children=[render_panel()],
        style={"padding":"12px", "display":"none"}
    )

    app.layout.children.append(panel_wrapper)

    # ====== CALLBACKS ======

    #toggle panel visibility based on host modality
    @app.callback(
        Output("mr-panel-wrapper", "style"),
        Input("modality-store", "data"),
        prevent_initial_call=False
    )
    def _toggle_panel(modality):
        """show/hide the panel based on modality."""
        base = {"padding":"12px"}
        if modality == "multi":
            return {**base, "display":"block"}
        return {**base, "display":"none"}

    #helper used by preview + exports
    def _build_wps(sel_data, xm,xp,ym,yp,zm,zp, radius, views, dwell):
        """return (waypoints, ids) for the requested growth, clamped to grid bounds."""
        if not sel_data or sel_data.get("id") is None:
            raise PreventUpdate
        anchor = int(sel_data["id"])
        req_ext = RegionExtents(int(xm or 0), int(xp or 0), int(ym or 0), int(yp or 0), int(zm or 0), int(zp or 0))

        #compute effective growth based on grid limits
        eff, _, _, _, _ = _clamped_growth_feedback(grid_ref, anchor, req_ext)
        eff_ext = RegionExtents(
            x_minus=eff['-X'], x_plus=eff['+X'],
            y_minus=eff['-Y'], y_plus=eff['+Y'],
            z_minus=eff['-Z'], z_plus=eff['+Z'],
        )

        #build IDs using the effective extents
        ids = get_region_ids(grid_ref, anchor, eff_ext)

        wps = generate_region_orbits(
            grid_ref, drone_ref, ids, generate_cube_orbit_fn,
            radius=float(radius or 0.6), n_views=int(views or 8), dwell_s=float(dwell or 1.5),
            floor_clear=0.0, ceil_clear=0.0
        )
        return wps, ids





    # --- summary/export builders  ---
    def compute_multi_summary(ids: List[int], waypoints: List, drone_config) -> dict:
        """
        build the same metrics the single summary shows (validate_mission_simple).
        Returns in stats:
        regions, total_waypoints, distance_m, total_time_s, battery_used_pct,
        battery_ok, battery_warning, distance_ok, feasible
        """
        stats = {
            "regions": len(ids),
            "total_waypoints": len(waypoints),
        }

        if not waypoints:
            stats.update({
                "distance_m": 0.0,
                "total_time_s": 0.0,
                "battery_used_pct": 0.0,
                "battery_ok": True,
                "battery_warning": False,
                "distance_ok": True,
                "feasible": True,
            })
            return {"title": "Multi-region flight summary", "stats": stats}

        fixed_overhead_s = 12.0
        wp_stop_s = 13.0
        speed_override = float(getattr(drone_config, "speed", 0.4) or 0.4)
        v_horiz = max(speed_override, 1e-6)
        v_vert = v_horiz
        battery_time_s = float(getattr(drone_config, "max_battery_time", 1380.0) or 1380.0)
        battery_drainage = 4.0
        ground_z = 0.0

        #horizontal cruise distance
        horiz = 0.0
        for a, b in zip(waypoints, waypoints[1:]):
            horiz += math.hypot(b.x - a.x, b.y - a.y)

        #vertical takeoff/landing legs
        takeoff = abs(waypoints[0].z - ground_z)
        landing = abs(waypoints[-1].z - ground_z)

        total_dist = horiz + takeoff + landing

        #times
        travel_time = (horiz / v_horiz) + ((takeoff + landing) / v_vert)
        hover_time  = sum((getattr(wp, "hold_time", 0.0) or 0.0) for wp in waypoints)
        stops_time  = wp_stop_s * len(waypoints)
        total_time  = float(travel_time + hover_time + stops_time + fixed_overhead_s)

        #battery usage (time-based) + extra drainage margin
        T_batt = max(float(battery_time_s), 1e-6)
        battery_used_pct = 100.0 * (total_time / T_batt) + float(battery_drainage)

        #limits
        max_dist = float(getattr(drone_config, "max_distance", float("inf")))
        distance_ok = (total_dist <= max_dist)

        #battery checks
        warn_thresh = float(getattr(drone_config, "battery_warning_threshold", 0.85) or 0.85)
        battery_ok = (battery_used_pct <= 100.0)
        battery_warning = (battery_used_pct >= (warn_thresh * 100.0))

        feasible = battery_ok and distance_ok

        stats.update({
            "distance_m": total_dist,
            "total_time_s": total_time,
            "battery_used_pct": battery_used_pct,
            "battery_ok": battery_ok,
            "battery_warning": battery_warning,
            "distance_ok": distance_ok,
            "feasible": feasible,
        })
        return {
            "title": "Multi-region flight summary",
            "stats": stats,
        }





    def build_multi_exports(ids: List[int], waypoints: List, drone_config) -> dict:
        csv_name = f"flight_region_{len(ids)}cubes.csv"
        txt_name = f"flight_region_{len(ids)}cubes.txt"
        dpt_name = f"flight_region_{len(ids)}cubes.dpt"
        return {
            "csv": dcc.send_string(_render_marvelmind_csv(waypoints), filename=csv_name),
            "txt": dcc.send_string(_render_txt(waypoints), filename=txt_name),
            "dpt": dcc.send_string(_render_dpt(waypoints, drone_config), filename=dpt_name),
        }

    #---preview callback: figure + status + stores---
    @app.callback(
        Output("mr-graph", "figure"),
        Output("mr-status", "children"),
        Output("multi-flight-summary-json", "data"),
        Output("multi-export-payloads", "data"),
        Input("mr-preview", "n_clicks"),
        State("modality-store", "data"),
        State("selected-id-store", "data"),
        State("mr-xm", "value"), State("mr-xp", "value"),
        State("mr-ym", "value"), State("mr-yp", "value"),
        State("mr-zm", "value"), State("mr-zp", "value"),
        State("mr-radius", "value"), State("mr-views", "value"), State("mr-dwell", "value"),
        prevent_initial_call=True
    )
    def _preview(n, modality, selected_data, xm,xp,ym,yp,zm,zp, radius, views, dwell):
        """handle preview click: compute ids, path, figure, status and populate stores."""
        if modality != "multi":
            raise PreventUpdate
        if not n or not selected_data or selected_data.get("id") is None:
            raise PreventUpdate

        wps, ids = _build_wps(selected_data, xm,xp,ym,yp,zm,zp, radius, views, dwell)
        fig = make_region_figure(
            grid_ref, ids, wps,
        )

        #feedback
        anchor_id = int(selected_data["id"])
        req_ext = RegionExtents(
            x_minus=int(xm or 0), x_plus=int(xp or 0),
            y_minus=int(ym or 0), y_plus=int(yp or 0),
            z_minus=int(zm or 0), z_plus=int(zp or 0),
        )
        eff, blocked_dirs, any_req, any_eff, req = _clamped_growth_feedback(grid_ref, anchor_id, req_ext)

        base = f"Nr. of cubes: {len(ids)} | Waypoints: {len(wps)}"
        notes = []

        if blocked_dirs:
            #per-direction detailed reason
            details = []
            for d in blocked_dirs:
                if eff[d] == 0:
                    details.append(f"{d}: requested {req[d]}, no room (using 0)")
                else:
                    details.append(f"{d}: requested {req[d]}, clipped to {eff[d]}")
            notes.append(" ; ".join(details))

        eff_pairs = [f"{d}={eff[d]}" for d in ['-X','+X','-Y','+Y','-Z','+Z'] if eff[d] > 0]
        if eff_pairs:
            notes.append("Applied growth: " + ", ".join(eff_pairs))

        if any_req and not any_eff:
            notes.append("Requested expansion did not take effect at this anchor. Try the opposite direction or reduce growth.")

        status = base if not notes else base + "  |  " + "  |  ".join(notes)



       
        summary = compute_multi_summary(ids, wps, drone_ref)
        exports = build_multi_exports(ids, wps, drone_ref)

        return fig, status, summary, exports

    # -------- exporters used above --------

    def _render_marvelmind_csv(waypoints: List) -> str:
        lines = ["index,x,y,z,HoldTime,GimblePitch"]
        for idx, wp in enumerate(waypoints, start=1):
            lines.append(f"{idx},{wp.x:.6f},{wp.y:.6f},{wp.z:.6f},{getattr(wp,'hold_time',0.0):.3f},{getattr(wp,'gimbal_pitch',0.0):.2f}")
        return "\n".join(lines)

    def _render_txt(waypoints: List) -> str:
        lines = ["# index x y z hold_time gimbal_pitch speed"]
        for i, wp in enumerate(waypoints, start=1):
            lines.append(f"{i} {wp.x:.6f} {wp.y:.6f} {wp.z:.6f} {getattr(wp,'hold_time',0.0):.3f} {getattr(wp,'gimbal_pitch',0.0):.2f} {getattr(wp,'speed',0.0):.3f}")
        return "\n".join(lines)

    def _render_dpt(waypoints: List, drone_config) -> str:
        lines = []
        lines.append(f"takeoff({getattr(drone_config,'min_altitude',1.0):.2f})")
        lines.append("pause(4.0)")
        lines.append(f"height({getattr(drone_config,'min_altitude',1.0):.2f})")
        lines.append("pause(1.0)")
        lines.append("waypoints_begin()")
        for i, wp in enumerate(waypoints):
            lines.append(f"W{i+1:02d}({wp.x:.2f},{wp.y:.2f},{wp.z:.2f})")
        lines.append("waypoints_end()")
        lines.append("pause(1.0)")
        lines.append("landing()")
        lines.append("pause(6.0)")
        lines.append("landing()")
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
            "set_min_speed(0.4)",
            "set_z_sliding_window(32)",
            "set_reverse_brake_time_c(0.67)",
            "set_reverse_brake_time_d(0.33)",
            "set_reverse_brake_time_e(0.00)",
        ])
        return "\n".join(lines)
