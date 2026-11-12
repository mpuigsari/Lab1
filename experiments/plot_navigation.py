#!/usr/bin/env python3
"""Plot the Lab 2 rosbags recorded by run_navigation.py as PNG figures, plus a summary table.

Run inside the container, with the workspace sourced:
    python3 plot_navigation.py                 # every case in nav_results/
    python3 plot_navigation.py nav4 csa3       # only the cases whose name contains one of these words
    python3 plot_navigation.py --robot 3       # robot shown in the acceleration figure (default 0)

Figures written to nav_figures/ (time 0 = the goal is given, i.e. the first /flock/path):
    <case>_navigation.png     centroid distance to the goal and centroid speed (mean velocity of all
                              robots), with the commanded flock speed: the swarm slows down and stops
    <case>_trajectories.png   robot paths on the map, colored where an obstacle rule was active,
                              planned centroid path, centroid path, start and goal
    <case>_safety.png         closest robot-obstacle and robot-robot distances, robots avoiding
    <case>_accelerations.png  magnitude of every rule for one robot
    comparison_maps.png       the four maps of the assignment (nav1-nav4)
    comparison_avoidance_<map>.png  potential field / steer-to-avoid / both (obstacle map, hard maze)
    summary.csv               one row per case, also printed: the numbers to tune the configs with

The centroid and its velocity use all the robots (stragglers included), from the odom positions.
The obstacle clearance is measured on the map grid (5 cm cells): about +-2.5 cm.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import yaml
from matplotlib.collections import LineCollection
from scipy import ndimage

# Style, palette and helpers of the Lab 1 figures (importing it also sets the matplotlib style)
from plot_experiments import (GRID, INK, INK_2, MUTED, SERIES, plt,  # noqa: E402
                              case_title, stamp_to_sec)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lab1_pkg'))  # GridMap from the source tree, no ROS needed
from lab1_pkg.utils.grid_map import GridMap  # noqa: E402

RESULTS_DIR = os.path.join(HERE, 'nav_results')
FIGURES_DIR = os.path.join(HERE, 'nav_figures')

ROBOT_RADIUS = 0.0375  # [m] sphero in Stage (size 0.075)
STEP = 0.1             # [s] common time grid of the analysis
VEL_WINDOW = 2         # grid steps on each side to differentiate the positions (0.4 s window)
AVOID_THRESHOLD = 1e-3  # [m/s^2] rule acceleration above which a robot is avoiding (as flock_markers_node)

# Colors of the avoidance states, as in RViz (flock_markers_node): normal, repulsion, steer only
NORMAL_COLOR, REPULSION_COLOR, STEER_COLOR = SERIES[0], '#eb6834', '#8e44c9'
RULE_COLORS = [('alignment', SERIES[0]), ('cohesion', SERIES[4]), ('separation', SERIES[2]),
               ('navigation', SERIES[3]), ('obstacle', REPULSION_COLOR), ('steer', STEER_COLOR)]

COMPARISONS = {
    'comparison_maps': ('Navigation on the four maps', ['nav1_empty', 'nav2_obstacles', 'nav3_simple_maze',
                                                       'nav4_hard_maze']),
    'comparison_avoidance_obstacles': ('Obstacle avoidance rules on the obstacle map: both / potential field only / '
                                       'steer-to-avoid only',
                                       ['nav2_obstacles', 'nav5_obstacles_potential_only', 'nav6_obstacles_steer_only']),
    'comparison_avoidance_hard_maze': ('Obstacle avoidance rules in the hard maze: both / potential field only / '
                                       'steer-to-avoid only',
                                       ['nav4_hard_maze', 'nav7_hard_maze_potential_only', 'nav8_hard_maze_steer_only']),
}


# ------------------------------------------------------------------------------------------- input
def read_bag(bag_dir, accel_robot):
    '''Raw series of a run:
        odom[robot]         (t, x, y)
        avoid[rule][robot]  (t, |a|) for the obstacle and steer rules of every robot
        accel[rule]         (t, |a|) for every rule of accel_robot
        flock[name]         (t, value) for cmd_speed, goal_distance, path_distance
        paths               [(t, (N, 2) waypoints)] every /flock/path
    Float64 and Path messages are timed with the recording time, which is sim time (--use-sim-time)'''
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    msg_types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    odom, accel, flock = defaultdict(list), defaultdict(list), defaultdict(list)
    avoid = {'obstacle': defaultdict(list), 'steer': defaultdict(list)}
    paths = []

    while reader.has_next():
        topic, data, bag_time = reader.read_next()
        parts = topic.strip('/').split('/')
        if parts[0] == 'flock':
            if parts[1] not in ('path', 'cmd_speed', 'goal_distance', 'path_distance'):
                continue
            msg = deserialize_message(data, msg_types[topic])
            if parts[1] == 'path':
                paths.append((bag_time * 1e-9, np.array([[p.pose.position.x, p.pose.position.y]
                                                         for p in msg.poses]).reshape(-1, 2)))
            else:
                flock[parts[1]].append((bag_time * 1e-9, msg.data))
            continue
        if not parts[0].startswith('robot_') or parts[1] == 'cmd_vel':
            continue  # /clock, commanded velocities (the actual ones come from odom)
        robot = int(parts[0].split('_')[1])
        if parts[1] == 'reynolds' and parts[2] not in avoid and robot != accel_robot:
            continue
        msg = deserialize_message(data, msg_types[topic])
        if parts[1] == 'odom':
            pos = msg.pose.pose.position
            odom[robot].append((stamp_to_sec(msg.header.stamp), pos.x, pos.y))
        elif parts[1] == 'reynolds':
            sample = (stamp_to_sec(msg.header.stamp), float(np.hypot(msg.vector.x, msg.vector.y)))
            if parts[2] in avoid:
                avoid[parts[2]][robot].append(sample)
            if robot == accel_robot:
                accel[parts[2]].append(sample)

    def to_array(samples):
        arr = np.array(sorted(samples))
        return arr[np.unique(arr[:, 0], return_index=True)[1]]  # one sample per time stamp

    return {'odom': {r: to_array(s) for r, s in odom.items()},
            'avoid': {rule: {r: to_array(s) for r, s in series.items()} for rule, series in avoid.items()},
            'accel': {rule: to_array(s) for rule, s in accel.items()},
            'flock': {name: to_array(s) for name, s in flock.items()},
            'paths': paths}


def map_yaml_path(map_name, maps_dir=None):
    if maps_dir is None:
        from ament_index_python.packages import get_package_share_directory
        maps_dir = os.path.join(get_package_share_directory('sphero_stage'), 'config', 'maps')
    return os.path.join(maps_dir, map_name, f'{map_name}.yaml')


# ------------------------------------------------------------------------------------------- analysis
def sample(series, t, default=0.0):
    '''Value of a (t, value) series at the times t (linear interpolation, default where it is missing)'''
    if series is None or len(series) == 0:
        return np.full(len(t), default)
    return np.interp(t, series[:, 0], series[:, 1], left=default, right=series[-1, 1])


def analyze(raw, params, grid):
    '''Everything the figures and the summary need, on a common time grid (time 0 = goal given)'''
    odom = raw['odom']
    robots = sorted(odom)
    t_abs = np.arange(max(o[0, 0] for o in odom.values()), min(o[-1, 0] for o in odom.values()), STEP)
    xy = np.stack([np.column_stack([np.interp(t_abs, odom[r][:, 0], odom[r][:, 1]),
                                    np.interp(t_abs, odom[r][:, 0], odom[r][:, 2])]) for r in robots],
                  axis=1)  # time, robot, xy

    # Navigation start: the first path (the goal is given); the goal is the end of the last path
    paths = [(t, p) for t, p in raw['paths'] if len(p)]
    path = paths[-1][1] if paths else None
    t0 = paths[0][0] if paths else t_abs[0]
    goal = path[-1] if path is not None else np.array([params['goal_x'], params['goal_y']], dtype=float)
    t = t_abs - t0

    centroid = xy.mean(axis=1)
    distance = np.linalg.norm(centroid - goal, axis=1)

    # Velocities: centered difference of the positions
    w = VEL_WINDOW
    vel = np.zeros_like(xy)
    vel[w:-w] = (xy[2 * w:] - xy[:-2 * w]) / (2 * w * STEP)
    vel[:w], vel[-w:] = vel[w], vel[-w - 1]
    centroid_speed = np.linalg.norm(vel.mean(axis=1), axis=1)  # |mean velocity| = speed of the centroid
    mean_speed = np.linalg.norm(vel, axis=2).mean(axis=1)

    flock = raw['flock']
    cmd_speed = sample(flock.get('cmd_speed'), t_abs, default=np.nan)
    path_distance = sample(flock.get('path_distance'), t_abs, default=np.nan) if 'path_distance' in flock else None

    # Goal reached as the flock controller decides it (goal_distance < goal_tol), else from the odom centroid
    goal_tol = float(params['goal_tol'])
    gd = flock.get('goal_distance')
    if gd is not None and len(gd) and (gd[:, 1] < goal_tol).any():
        t_goal = gd[np.argmax(gd[:, 1] < goal_tol), 0] - t0
    elif (distance[t >= 0] < goal_tol).any():
        t_goal = t[t >= 0][np.argmax(distance[t >= 0] < goal_tol)]
    else:
        t_goal = None
    slowing = float(params['v_max']) / float(params['k'])  # the flock slows down over the last v_max / k meters
    if path_distance is not None and (path_distance < slowing).any():
        t_slow = t[np.argmax(path_distance < slowing)]
    else:
        t_slow = t[np.argmax(distance < slowing)] if (distance < slowing).any() else None

    # Safety: clearance of every robot to the obstacles (map grid), distance between robots
    edt = ndimage.distance_transform_edt(grid.occupancy == 0) * grid.resolution
    cols, rows = np.floor((xy - grid.origin) / grid.resolution).astype(int).transpose(2, 0, 1)
    rows = np.clip(rows, 0, grid.height - 1)
    cols = np.clip(cols, 0, grid.width - 1)
    clearance = edt[rows, cols] - 0.5 * grid.resolution  # time, robot
    pair = np.linalg.norm(xy[:, :, None] - xy[:, None], axis=3)
    pair[:, np.arange(len(robots)), np.arange(len(robots))] = np.inf

    # Avoidance state of every robot: 2 repulsion active, 1 only steer-to-avoid, 0 none
    repulsion = np.stack([sample(raw['avoid']['obstacle'].get(r), t_abs) > AVOID_THRESHOLD for r in robots], axis=1)
    steer = np.stack([sample(raw['avoid']['steer'].get(r), t_abs) > AVOID_THRESHOLD for r in robots], axis=1)
    state = np.where(repulsion, 2, np.where(steer, 1, 0))

    final = xy[-1]
    final_centroid = final.mean(axis=0)
    after = t >= 0
    return {
        'robots': robots, 't': t, 'xy': xy, 'centroid': centroid, 'goal': goal, 'path': path,
        'distance': distance, 'path_distance': path_distance, 'centroid_speed': centroid_speed,
        'mean_speed': mean_speed, 'cmd_speed': cmd_speed, 't_goal': t_goal, 't_slow': t_slow,
        'clearance': clearance, 'min_clearance': clearance.min(axis=1), 'min_pair': pair.min(axis=(1, 2)),
        'state': state, 'accel': raw['accel'], 't0': t0,
        'summary': {
            'goal_time_s': t_goal,
            'path_length_m': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if path is not None else None,
            'final_error_m': float(distance[-1]),
            'final_spread_m': float(np.linalg.norm(final - final_centroid, axis=1).mean()),
            'min_clearance_m': float(clearance[after].min()),
            'contact_time_s': float(STEP * (clearance[after].min(axis=1) < ROBOT_RADIUS).sum()),
            'min_robot_distance_m': float(pair[after].min()),
            'stragglers': int((np.linalg.norm(final - final_centroid, axis=1) > float(params['max_dist'])).sum()),
            'peak_centroid_speed': float(centroid_speed[after].max()),
            'repulsion_active': float((state[after] == 2).any(axis=1).mean()),
            'steer_active': float((state[after] >= 1).any(axis=1).mean()),
        },
    }


# ------------------------------------------------------------------------------------------- figures
def describe(params, wrap=False):
    '''The parameters of a case, on one line (or two with wrap)'''
    flock = (f"{params['map_name']},  N = {params['number_robots']},  ca = {params['ca']:g},  cc = {params['cc']:g},  "
             f"cs = {params['cs']:g},  cn = {params['cn']:g}")
    rest = f"co = {params['co']:g},  csa = {params['csa']:g},  v_max = {params['v_max']:g} m/s,  k = {params['k']:g} 1/s"
    return f'{flock}\n{rest}' if wrap else f'{flock},  {rest}'


def legend_above(ax, ncol):
    '''Legend in a row above the panel, so it never hides a curve'''
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.3)


def time_window(a, settle=10.0):
    '''Time axis: from the goal being given to shortly after it is reached'''
    end = a['t'][-1] if a['t_goal'] is None else min(a['t'][-1], a['t_goal'] + settle)
    return min(0.0, a['t'][0]), end


def mark_phases(ax, a, label=False):
    '''Shade the arrival phase (last v_max / k meters) and mark the goal'''
    if a['t_slow'] is not None:
        end = a['t_goal'] if a['t_goal'] is not None else a['t'][-1]
        ax.axvspan(a['t_slow'], end, color=GRID, alpha=0.6, lw=0, zorder=0,
                   label='arrival: slowing down' if label else None)
    if a['t_goal'] is not None:
        ax.axvline(a['t_goal'], color=INK_2, lw=1.0, ls=(0, (4, 2)),
                   label=f"goal reached ({a['t_goal']:.1f} s)" if label else None)


def plot_navigation(name, params, a, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    ax = axes[0]
    ax.plot(a['t'], a['distance'], color=SERIES[0], lw=1.8, label='straight line')
    if a['path_distance'] is not None:
        ax.plot(a['t'], a['path_distance'], color=SERIES[1], lw=1.3, ls=(0, (5, 2)), label='along the path')
    ax.axhline(float(params['goal_tol']), color=MUTED, lw=0.8, ls=':', label='goal tolerance')
    mark_phases(ax, a, label=True)
    ax.set_ylabel('centroid distance\nto the goal [m]')
    ax.set_ylim(bottom=0)
    legend_above(ax, 5)

    ax = axes[1]
    ax.plot(a['t'], a['centroid_speed'], color=SERIES[0], lw=1.8, label='centroid (mean velocity of all robots)')
    ax.plot(a['t'], a['mean_speed'], color=SERIES[2], lw=1.0, alpha=0.8, label='mean robot speed')
    ax.plot(a['t'], a['cmd_speed'], color=INK, lw=1.0, ls=(0, (3, 2)), drawstyle='steps-post',
            label='commanded flock speed')
    ax.axhline(float(params['v_max']), color=MUTED, lw=0.8, ls=':', label='v_max')
    mark_phases(ax, a)
    ax.set_ylabel('speed [m/s]')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('time since the goal was given [s]')
    legend_above(ax, 4)
    axes[0].set_xlim(*time_window(a))
    fig.suptitle(f'{case_title(name)}: centroid distance to the goal and centroid velocity\n{describe(params)}',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def draw_map(ax, grid):
    lo, hi = grid.bounds()
    blocked = (grid.occupancy != 0).astype(float)
    ax.imshow(np.ma.masked_where(blocked == 0, blocked), cmap='Greys', vmin=0, vmax=1.4, origin='lower',
              extent=(lo[0], hi[0], lo[1], hi[1]), interpolation='nearest', zorder=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')


def plot_trajectories(name, params, a, grid, path):
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    draw_map(ax, grid)
    colors = np.array([NORMAL_COLOR, STEER_COLOR, REPULSION_COLOR])
    for k in range(len(a['robots'])):
        pts = a['xy'][:, k]
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(segments, colors=colors[a['state'][:-1, k]], linewidths=1.1,
                                         alpha=0.85, zorder=2, capstyle='round'))
        ax.plot(*pts[0], 'o', ms=4.5, mfc='white', mec=INK_2, mew=0.9, zorder=4)
        ax.plot(*pts[-1], 'o', ms=4.5, color=INK_2, mec='white', mew=0.6, zorder=4)
    if a['path'] is not None:
        ax.plot(a['path'][:, 0], a['path'][:, 1], color=SERIES[2], lw=1.4, ls=(0, (4, 2)), zorder=3)
    ax.plot(a['centroid'][:, 0], a['centroid'][:, 1], color=INK, lw=1.4, zorder=3)
    ax.plot(*a['goal'], marker='*', ms=15, color=SERIES[3], mec=INK, mew=0.6, zorder=5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(False)

    handles = [plt.Line2D([], [], color=NORMAL_COLOR, lw=1.5), plt.Line2D([], [], color=STEER_COLOR, lw=1.5),
               plt.Line2D([], [], color=REPULSION_COLOR, lw=1.5), plt.Line2D([], [], color=SERIES[2], lw=1.4, ls=(0, (4, 2))),
               plt.Line2D([], [], color=INK, lw=1.4),
               plt.Line2D([], [], ls='', marker='o', ms=5, mfc='white', mec=INK_2),
               plt.Line2D([], [], ls='', marker='o', ms=5, color=INK_2),
               plt.Line2D([], [], ls='', marker='*', ms=11, color=SERIES[3], mec=INK, mew=0.6)]
    labels = ['robot path', 'steer-to-avoid active', 'potential field active', 'planned centroid path',
              'centroid path', 'start', 'end', 'goal']
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -0.08), ncol=4, fontsize=7.5)
    fig.suptitle(f'{case_title(name)}: trajectories\n{describe(params, wrap=True)}', x=0.02, ha='left', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path)
    plt.close(fig)


def plot_safety(name, params, a, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True)
    ax = axes[0]
    ax.plot(a['t'], a['min_clearance'], color=SERIES[0], lw=1.5, label='closest robot (center) to an obstacle')
    ax.axhline(ROBOT_RADIUS, color=SERIES[4], lw=1.0, label=f'contact ({ROBOT_RADIUS} m, robot radius)')
    ax.axhline(float(params['obs_range']), color=REPULSION_COLOR, lw=0.8, ls=':', label='potential field range')
    ax.set_ylabel('distance [m]')
    ax.set_ylim(0, max(0.6, 1.5 * float(params['obs_range'])))
    legend_above(ax, 3)

    ax = axes[1]
    ax.plot(a['t'], a['min_pair'], color=SERIES[0], lw=1.5, label='closest pair of robots (centers)')
    ax.axhline(2 * ROBOT_RADIUS, color=SERIES[4], lw=1.0, label=f'contact ({2 * ROBOT_RADIUS} m)')
    ax.set_ylabel('distance [m]')
    ax.set_ylim(bottom=0)
    legend_above(ax, 2)

    ax = axes[2]
    ax.step(a['t'], (a['state'] >= 1).sum(axis=1), where='post', color=STEER_COLOR, lw=1.2,
            label='any obstacle rule active')
    ax.step(a['t'], (a['state'] == 2).sum(axis=1), where='post', color=REPULSION_COLOR, lw=1.2,
            label='potential field active')
    ax.set_ylabel('robots avoiding')
    ax.set_ylim(0, len(a['robots']) + 0.5)
    legend_above(ax, 2)
    ax.set_xlabel('time since the goal was given [s]')
    for ax in axes:
        mark_phases(ax, a)
    axes[0].set_xlim(*time_window(a))
    fig.suptitle(f'{case_title(name)}: safety\n{describe(params)}', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_accelerations(name, params, a, robot, path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for rule, color in RULE_COLORS + [('total', INK)]:
        if rule in a['accel']:
            s = a['accel'][rule]
            ax.plot(s[:, 0] - a['t0'], s[:, 1], color=color, lw=1.6 if rule == 'total' else 1.1, label=rule)
    mark_phases(ax, a)
    ax.axhline(float(params['max_acc']), color=MUTED, lw=0.8, ls=':', label='max_acc')
    ax.set_xlim(*time_window(a))
    ax.set_ylim(bottom=0)
    ax.set_xlabel('time since the goal was given [s]')
    ax.set_ylabel('magnitude [m/s²]')
    legend_above(ax, 8)
    fig.suptitle(f'{case_title(name)}: acceleration of each rule (after the allocation), robot {robot}\n'
                 f'{describe(params)}', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(title, runs, path):
    '''runs: {name: analysis}. Distance to the goal, centroid speed, closest obstacle and closest robot'''
    fig, axes = plt.subplots(4, 1, figsize=(10, 9.5), sharex=True)
    panels = [('distance', 'centroid distance\nto the goal [m]'), ('centroid_speed', 'centroid speed [m/s]'),
              ('min_clearance', 'closest robot to\nan obstacle [m]'), ('min_pair', 'closest pair\nof robots [m]')]
    colors = SERIES + [INK, MUTED]
    end = 0.0
    for (name, a), color in zip(runs.items(), colors):
        for ax, (key, _) in zip(axes, panels):
            ax.plot(a['t'], a[key], color=color, lw=1.4, label=case_title(name))
        if a['t_goal'] is not None:
            axes[0].plot(a['t_goal'], float(a['distance'][np.searchsorted(a['t'], a['t_goal'])]), 'o',
                         color=color, ms=5)
        end = max(end, time_window(a)[1])
    for ax, (_, label) in zip(axes, panels):
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[2].set_ylim(0, min(axes[2].get_ylim()[1], 1.0))  # open space would flatten the close passes
    axes[2].axhline(ROBOT_RADIUS, color=SERIES[4], lw=0.8, ls=':')
    axes[3].axhline(2 * ROBOT_RADIUS, color=SERIES[4], lw=0.8, ls=':')
    axes[0].set_xlim(0, end)
    axes[-1].set_xlabel('time since the goal was given [s]  (dot: goal reached; dotted: contact; '
                        'obstacle distance capped at 1 m)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right', ncol=2, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(title, x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main
def fmt(value):
    if value is None:
        return '-'
    return f'{value:.2f}' if isinstance(value, float) else str(value)


def print_summary(rows):
    keys = list(next(iter(rows.values())).keys())
    short = {'goal_time_s': 'goal[s]', 'path_length_m': 'path[m]', 'final_error_m': 'err[m]',
             'final_spread_m': 'spread[m]', 'min_clearance_m': 'clear[m]', 'contact_time_s': 'contact[s]',
             'min_robot_distance_m': 'robots[m]', 'stragglers': 'strag', 'peak_centroid_speed': 'vmax[m/s]',
             'repulsion_active': 'rep.frac', 'steer_active': 'avoid.frac'}
    width = max(len(n) for n in rows) + 2
    widths = {k: max(len(short.get(k, k)), *(len(fmt(row[k])) for row in rows.values())) + 2 for k in keys}
    print('case'.ljust(width) + ''.join(short.get(k, k).rjust(widths[k]) for k in keys))
    for name, row in rows.items():
        print(name.ljust(width) + ''.join(fmt(row[k]).rjust(widths[k]) for k in keys))


def process_case(name, raw, params, grid, robot, figures_dir):
    a = analyze(raw, params, grid)
    prefix = os.path.join(figures_dir, name)
    plot_navigation(name, params, a, f'{prefix}_navigation.png')
    plot_trajectories(name, params, a, grid, f'{prefix}_trajectories.png')
    plot_safety(name, params, a, f'{prefix}_safety.png')
    if a['accel']:
        plot_accelerations(name, params, a, robot, f'{prefix}_accelerations.png')
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', nargs='*', help='only the cases whose name contains one of these words')
    parser.add_argument('--robot', type=int, default=0, help='robot of the acceleration figure')
    parser.add_argument('--maps-dir', default=None, help='sphero_stage maps folder (default: from the ROS install)')
    args = parser.parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    analyses, rows, grids = {}, {}, {}
    for name in sorted(os.listdir(RESULTS_DIR)):
        if args.cases and not any(word in name for word in args.cases):
            continue
        bag_dir = os.path.join(RESULTS_DIR, name, 'bag')
        if not os.path.isfile(os.path.join(bag_dir, 'metadata.yaml')):
            continue
        with open(os.path.join(RESULTS_DIR, name, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)['/**']['ros__parameters']
        map_name = params['map_name']
        if map_name not in grids:
            grids[map_name] = GridMap.from_yaml(map_yaml_path(map_name, args.maps_dir))
        raw = read_bag(bag_dir, args.robot)
        if not raw['odom']:
            print(f'{name}: no odometry in the bag, skipped')
            continue
        analyses[name] = process_case(name, raw, params, grids[map_name], args.robot, FIGURES_DIR)
        rows[name] = {'map': map_name, **analyses[name]['summary']}

    for figure, (title, names) in COMPARISONS.items():
        runs = {name: analyses[name] for name in names if name in analyses}
        if len(runs) > 1:
            plot_comparison(title, runs, os.path.join(FIGURES_DIR, f'{figure}.png'))

    if rows:
        print_summary(rows)
        with open(os.path.join(FIGURES_DIR, 'summary.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['case', *next(iter(rows.values())).keys()])
            writer.writeheader()
            for name, row in rows.items():
                writer.writerow({'case': name, **row})
    print(f'figures written to {FIGURES_DIR}')


if __name__ == '__main__':
    main()
