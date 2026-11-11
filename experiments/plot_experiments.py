#!/usr/bin/env python3
"""Plot the rosbags recorded by run_experiments.py as PNG figures.

Run inside the container, with the workspace sourced:
    python3 plot_experiments.py             # every case in results/
    python3 plot_experiments.py --robot 3   # robot shown in the acceleration figure (default 0)

Figures written to figures/:
    <case>_velocities.png     commanded vs actual velocity of every robot, x and y
    <case>_accelerations.png  acceleration of each Reynolds rule and the total, for one robot
    <case>_trajectories.png   path of every robot
    comparison_<group>.png    cluster radius, nearest neighbor distance and speed of one group of
                              cases (weights, single rules, agents, range and fov, map), each
                              with the baseline as reference (see COMPARISONS)

Time axes stop shortly after the last visible change: once every curve stays within VISIBLE of
the plot's y range from its final value, the rest of the run adds nothing to the figure.

The actual velocity is the derivative of the odom position. Stage reports the commanded velocity
in the odom twist even while a robot is blocked by a collision, so the twist would hide stalls.
"""
import argparse
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # write files only, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rosbag2_py  # noqa: E402
import yaml  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
FIGURES_DIR = os.path.join(HERE, 'figures')

# Categorical slots in fixed order, plus chart ink (validated reference palette, light mode)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, INK_2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
RULES = [('alignment', 'ca', SERIES[0]), ('cohesion', 'cc', SERIES[1]),
         ('separation', 'cs', SERIES[2])]
# Cases compared in each comparison figure (by case number); the baseline is the reference in both
COMPARISONS = {
    'comparison_weights': ['case1', 'case2', 'case3', 'case10', 'case11'],
    'comparison_single_rules': ['case1', 'case7', 'case8', 'case9'],
    'comparison_agents': ['case1', 'case6', 'case12', 'case13'],
    'comparison_range_fov': ['case1', 'case4', 'case5', 'case14', 'case15', 'case16'],
    'comparison_map': ['case1', 'case3', 'case17', 'case18'],
}
COMPARISON_TITLES = {
    'comparison_weights': 'Swarm metrics: rule weights',
    'comparison_single_rules': 'Swarm metrics: each rule on its own vs the baseline',
    'comparison_agents': 'Swarm metrics: number of robots',
    'comparison_range_fov': 'Swarm metrics: neighborhood radius and field of view',
    'comparison_map': 'Swarm metrics: map size (3 x 3 m vs 10 x 10 m)',
}
DIFF_WINDOW = 5  # odom samples (about 50 ms) used to differentiate the position
VISIBLE = 0.03  # a change smaller than this share of the y range is not visible in a plot
MIN_WINDOW = 10.0  # [s] shortest time axis

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200,
    'font.family': 'sans-serif', 'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK_2,
    'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.frameon': False, 'legend.fontsize': 8, 'text.color': INK,
})


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def read_bag(bag_dir, accel_robot):
    '''Return odom positions, commanded velocities and the rule accelerations of one robot'''
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    msg_types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    odom, cmd, accel = defaultdict(list), defaultdict(list), defaultdict(list)

    while reader.has_next():
        topic, data, bag_time = reader.read_next()
        parts = topic.strip('/').split('/')
        if not parts[0].startswith('robot_'):
            continue  # /clock
        robot = int(parts[0].split('_')[1])
        if parts[1] == 'reynolds' and robot != accel_robot:
            continue
        msg = deserialize_message(data, msg_types[topic])
        if parts[1] == 'odom':
            pos = msg.pose.pose.position
            odom[robot].append((stamp_to_sec(msg.header.stamp), pos.x, pos.y))
        elif parts[1] == 'cmd_vel':
            # Twist has no header: use the recording time, which is sim time (--use-sim-time)
            cmd[robot].append((bag_time * 1e-9, msg.linear.x, msg.linear.y))
        elif parts[1] == 'reynolds':
            accel[parts[2]].append((stamp_to_sec(msg.header.stamp), msg.vector.x, msg.vector.y))

    def to_arrays(samples):
        arr = np.array(sorted(samples))
        return arr[np.unique(arr[:, 0], return_index=True)[1]]  # one sample per time stamp

    return ({r: to_arrays(s) for r, s in odom.items()},
            {r: to_arrays(s) for r, s in cmd.items()},
            {name: to_arrays(s) for name, s in accel.items()})


def actual_velocity(odom):
    '''Velocity from the change of position over DIFF_WINDOW samples: (t, vx, vy)'''
    t, xy = odom[:, 0], odom[:, 1:3]
    k = DIFF_WINDOW
    dt = (t[k:] - t[:-k])[:, None]
    return np.column_stack([(t[k:] + t[:-k]) / 2, (xy[k:] - xy[:-k]) / dt])


def active_until(curves):
    '''Last time any curve still differs visibly from its final value; curves share one y axis'''
    values = np.concatenate([y for _, y in curves])
    tolerance = VISIBLE * (values.max() - values.min())
    end = 0.0
    for t, y in curves:
        moving = np.nonzero(np.abs(y - y[-1]) > tolerance)[0]
        if len(moving):
            end = max(end, t[moving[-1]])
    return end


def window_end(active, duration):
    '''End of the time axis: the last visible change plus a short margin'''
    return min(duration, max(MIN_WINDOW, active + max(2.0, 0.15 * active)))


def active_rules(accel, params):
    '''Lines of the acceleration figure: (topic, label, color).

    A rule with zero weight is a flat line at zero and is left out. With a single rule left,
    the total is the same curve, so only that rule is drawn.
    '''
    rules = [(rule, rule, color) for rule, weight, color in RULES
             if params[weight] != 0 and rule in accel]
    if len(rules) == 1:
        rule, _, color = rules[0]
        return [(rule, f'{rule} = total', color)]
    return rules + [('total', 'total', INK)]


def velocity_window(vel, cmd, duration):
    '''Time axis of the velocity figure: until the actual and commanded velocities settle'''
    curves = [(v[:, 0], v[:, col]) for v in vel.values() for col in (1, 2)]
    curves += [(c[:, 0], c[:, col]) for c in cmd.values() for col in (1, 2)]
    return window_end(active_until(curves), duration)


def acceleration_window(accel, duration):
    '''Time axis of the acceleration figure: until the total acceleration settles.

    In a settled cluster the single rule terms can keep drifting slowly while they cancel
    each other, which is not activity, so they do not decide the window.
    '''
    total = accel['total']
    components = [(total[:, 0], total[:, col]) for col in (1, 2)]
    magnitude = [(total[:, 0], np.hypot(total[:, 1], total[:, 2]))]
    return window_end(max(active_until(components), active_until(magnitude)), duration)


def crop_note(fig, t_end, duration):
    '''Say in the figure that the time axis is shorter than the run'''
    if t_end < duration - 0.5:
        fig.text(0.99, 0.005, f'Showing the first {t_end:.0f} s of {duration:.0f} s: '
                 'after that no curve changes visibly.', ha='right', va='bottom',
                 color=MUTED, fontsize=8)


def describe(params):
    '''One line with the parameters of a case'''
    text = (f"N = {params['number_robots']},  ca = {params['ca']:g},  cc = {params['cc']:g},  "
            f"cs = {params['cs']:g},  range = {params['max_range']:g} m,  fov = {params['fov']:g}°")
    if 'max_acc' in params:  # runs made before the acceleration limit existed have no limit
        text += f",  max acc = {params['max_acc']:g} m/s²"
    return text


def case_title(name):
    number, _, rest = name.partition('_')
    return f"{number.replace('case', 'Case ')}: {rest.replace('_', ' ')}"


def plot_velocities(name, params, vel, cmd, t_end, duration, path):
    robots = sorted(vel)
    ncols = 2 if len(robots) <= 10 else 4
    nrows = math.ceil(len(robots) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 1.55 * nrows + 1.2),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(robots):]:
        ax.set_visible(False)
    for ax, robot in zip(axes.flat, robots):
        for col, color in ((1, SERIES[0]), (2, SERIES[1])):
            ax.plot(vel[robot][:, 0], vel[robot][:, col], color=color, lw=2.4, alpha=0.45,
                    solid_capstyle='round')
            if robot in cmd:
                ax.plot(cmd[robot][:, 0], cmd[robot][:, col], color=color, lw=1.0,
                        ls=(0, (3, 2)), drawstyle='steps-post')
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_title(f'robot {robot}', loc='left', pad=2)
    axes[0, 0].set_xlim(0, t_end)
    for ax in axes[-1]:
        ax.set_xlabel('simulated time [s]')
    for ax in axes[:, 0]:
        ax.set_ylabel('velocity [m/s]')

    handles = [plt.Line2D([], [], color=SERIES[0], lw=1.0, ls=(0, (3, 2))),
               plt.Line2D([], [], color=SERIES[0], lw=2.4, alpha=0.45),
               plt.Line2D([], [], color=SERIES[1], lw=1.0, ls=(0, (3, 2))),
               plt.Line2D([], [], color=SERIES[1], lw=2.4, alpha=0.45)]
    fig.legend(handles, ['vx commanded', 'vx actual', 'vy commanded', 'vy actual'],
               loc='upper right', ncol=4, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f'{case_title(name)}: commanded vs actual velocity\n{describe(params)}',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    crop_note(fig, t_end, duration)
    fig.savefig(path)
    plt.close(fig)


def plot_accelerations(name, params, accel, robot, t_end, duration, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[1].sharey(axes[0])  # same scale for x and y, so a near zero component looks near zero
    labels = ('x component [m/s²]', 'y component [m/s²]', 'magnitude [m/s²]')
    for ax, label, col in zip(axes, labels, (1, 2, None)):
        for rule, legend, color in active_rules(accel, params):
            a = accel[rule]
            values = np.hypot(a[:, 1], a[:, 2]) if col is None else a[:, col]
            ax.plot(a[:, 0], values, color=color, lw=1.6 if rule == 'total' else 1.3, label=legend)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_ylabel(label)
    axes[0].set_xlim(0, t_end)
    axes[-1].set_xlabel('simulated time [s]')
    axes[0].legend(loc='upper right', ncol=4, bbox_to_anchor=(1.0, 1.25))
    fig.suptitle(f'{case_title(name)}: acceleration of each rule, robot {robot}\n{describe(params)}',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    crop_note(fig, t_end, duration)
    fig.savefig(path)
    plt.close(fig)


def plot_trajectories(name, params, odom, robot, path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for r in sorted(odom):
        xy = odom[r][:, 1:3]
        color, width, z = (SERIES[1], 1.8, 3) if r == robot else (SERIES[0], 1.2, 2)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=width, alpha=0.8, zorder=z)
        ax.plot(*xy[0], 'o', ms=8, mfc='white', mec=color, mew=1.5, zorder=z + 1)
        ax.plot(*xy[-1], 'o', ms=8, color=color, mec='white', mew=1.0, zorder=z + 1)
    ax.set_aspect('equal')
    ax.margins(0.08)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    handles = [plt.Line2D([], [], color=SERIES[0], lw=1.2),
               plt.Line2D([], [], color=SERIES[1], lw=1.8),
               plt.Line2D([], [], ls='', marker='o', ms=8, mfc='white', mec=MUTED, mew=1.5),
               plt.Line2D([], [], ls='', marker='o', ms=8, color=MUTED)]
    ax.legend(handles, ['robot path', f'robot {robot}', 'start', 'end'],
              loc='upper left', bbox_to_anchor=(0, 1.08), ncol=4)
    fig.suptitle(f'{case_title(name)}: trajectories\n{describe(params)}',
                 x=0.02, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def swarm_metrics(odom, step=0.1):
    '''Cluster radius, mean nearest neighbor distance and mean speed on a common time grid'''
    end = min(o[-1, 0] for o in odom.values())
    t = np.arange(max(o[0, 0] for o in odom.values()), end, step)
    xy = np.stack([np.column_stack([np.interp(t, o[:, 0], o[:, 1]), np.interp(t, o[:, 0], o[:, 2])])
                   for o in odom.values()], axis=1)  # time, robot, xy
    radius = np.linalg.norm(xy - xy.mean(axis=1, keepdims=True), axis=2).mean(axis=1)
    dist = np.linalg.norm(xy[:, :, None] - xy[:, None], axis=3)
    dist[:, np.arange(xy.shape[1]), np.arange(xy.shape[1])] = np.inf
    nearest = dist.min(axis=2).mean(axis=1)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=2).mean(axis=1) / step
    return t, radius, nearest, np.append(speed, speed[-1])


def plot_comparison(metrics, title, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    labels = ('cluster radius [m]', 'mean nearest neighbor\ndistance [m]', 'mean speed [m/s]')
    for (name, (t, *values)), color in zip(metrics.items(), SERIES):
        for ax, value in zip(axes, values):
            ax.plot(t, value, color=color, lw=1.5, label=case_title(name))
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)

    # Each panel has its own y axis, so each one decides separately when it stops changing
    duration = max(t[-1] for t, *_ in metrics.values())
    active = max(active_until([(t, values[panel]) for t, *values in metrics.values()])
                 for panel in range(3))
    t_end = window_end(active, duration)
    axes[0].set_xlim(0, t_end)
    axes[-1].set_xlabel('simulated time [s]')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right', ncol=2,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f'{title}\ncluster radius: mean distance of the robots to their centroid',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0.015, 1, 0.92))
    crop_note(fig, t_end, duration)
    fig.savefig(path)
    plt.close(fig)
    return t_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=int, default=0, help='robot of the acceleration figure')
    args = parser.parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metrics = {}
    for name in sorted(os.listdir(RESULTS_DIR)):
        bag_dir = os.path.join(RESULTS_DIR, name, 'bag')
        if not os.path.isfile(os.path.join(bag_dir, 'metadata.yaml')):
            continue
        with open(os.path.join(RESULTS_DIR, name, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)['/**']['ros__parameters']
        odom, cmd, accel = read_bag(bag_dir, args.robot)
        vel = {r: actual_velocity(o) for r, o in odom.items()}
        duration = max(o[-1, 0] for o in odom.values())
        prefix = os.path.join(FIGURES_DIR, name)
        t_vel = velocity_window(vel, cmd, duration)
        plot_velocities(name, params, vel, cmd, t_vel, duration, f'{prefix}_velocities.png')
        if 'total' in accel:
            t_acc = acceleration_window(accel, duration)
            plot_accelerations(name, params, accel, args.robot, t_acc, duration,
                               f'{prefix}_accelerations.png')
            accel_note = f'{t_acc:.0f} s (accelerations)'
        else:
            accel_note = f'NO acceleration data for robot {args.robot}, figure skipped'
        plot_trajectories(name, params, odom, args.robot, f'{prefix}_trajectories.png')
        metrics[name] = swarm_metrics(odom)
        silent = sorted(set(odom) - set(cmd))
        print(f'{name}: {len(odom)} robots, time axis {t_vel:.0f} s (velocities), {accel_note} '
              f'of {duration:.0f} s' + (f'; robots without commands: {silent}' if silent else ''))

    for figure, numbers in COMPARISONS.items():
        selected = {name: metrics[name] for number in numbers
                    for name in metrics if name.split('_')[0] == number}
        if selected:
            t_end = plot_comparison(selected, COMPARISON_TITLES[figure],
                                    os.path.join(FIGURES_DIR, f'{figure}.png'))
            print(f'{figure}: time axis {t_end:.0f} s')
    print(f'figures written to {FIGURES_DIR}')


if __name__ == '__main__':
    main()
