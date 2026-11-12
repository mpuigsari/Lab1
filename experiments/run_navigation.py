#!/usr/bin/env python3
"""Run the Lab 2 navigation + obstacle avoidance cases in Stage and record one rosbag per case.

Run inside the container, with the workspace sourced:
    python3 run_navigation.py                         # all cases
    python3 run_navigation.py nav4                    # only these cases (number or full name)
    python3 run_navigation.py nav4 --set csa=3.0 steer_time=2.0 --tag csa3   # tuning run
    python3 run_navigation.py nav2 --set circle.radius=0.5                   # nested keys with dots

Each case starts from one of the per-map config files in lab1_pkg/config (read from the source tree,
so edits apply without rebuilding) plus its overrides, and gets a folder in nav_results/ with its
params file, the launch log and the bag. A run stops `--settle` simulated seconds after the flock
reaches the goal, or at `--duration`. Then plot with plot_navigation.py.
Stage runs with its window open: only then does it keep simulated time in step with real time.
"""
import argparse
import copy
import os
import shutil
import subprocess
import time

# Same DDS isolation as the Lab 1 runner (set when it is imported) and its process helpers
from run_experiments import stop  # noqa: E402  (sets ROS_DOMAIN_ID / ROS_LOCALHOST_ONLY first)

import rclpy  # noqa: E402
import yaml  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, '..', 'lab1_pkg', 'config')
RESULTS_DIR = os.path.join(HERE, 'nav_results')
TOPICS_REGEX = (r'^/clock$|^/robot_[0-9]+/(odom|cmd_vel|reynolds/[a-z_]+)$'
                r'|^/flock/(path|cmd_vel|cmd_speed|goal_distance|path_distance)$')

# name: (config file in lab1_pkg/config, overrides)
CASES = {
    # One case per map of the assignment (Fig. 1 a-d), with the tuned config of that map
    'nav1_empty': ('lab2_empty.yaml', {}),
    'nav2_obstacles': ('lab2_obstacles.yaml', {}),
    'nav3_simple_maze': ('lab2_simple_maze.yaml', {}),
    'nav4_hard_maze': ('lab2_hard_maze.yaml', {}),
    # Obstacle avoidance rules on their own, to compare with both together (nav2, nav4)
    'nav5_obstacles_potential_only': ('lab2_obstacles.yaml', {'csa': 0.0}),
    'nav6_obstacles_steer_only': ('lab2_obstacles.yaml', {'co': 0.0}),
    'nav7_hard_maze_potential_only': ('lab2_hard_maze.yaml', {'csa': 0.0}),
    'nav8_hard_maze_steer_only': ('lab2_hard_maze.yaml', {'co': 0.0}),
}


def set_key(params, dotted_key, value):
    '''params['a']['b'] = value for 'a.b' (e.g. circle.radius)'''
    *parents, key = dotted_key.split('.')
    for parent in parents:
        params = params[parent]
    if key not in params:
        print(f'  warning: {dotted_key} is not in the config file, adding it')
    params[key] = copy.deepcopy(value)


def write_params(case_dir, config, overrides):
    '''Write the params file of a case: the config file with the case overrides'''
    with open(os.path.join(CONFIG_DIR, config), 'r') as f:
        params = yaml.safe_load(f)
    ros_params = params['/**']['ros__parameters']
    for key, value in overrides.items():
        set_key(ros_params, key, value)
    path = os.path.join(case_dir, 'params.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(params, f, sort_keys=False)
    return path, ros_params


def wait_for_goal(node, duration, settle, goal_tol, timeout):
    '''Block until settle seconds after the centroid comes within goal_tol of the goal (as the flock
    controller decides it), or until duration. Return (sim time, sim time of the goal or None, wall time)'''
    state = {'now': 0.0, 'goal': None}

    def on_clock(msg):
        state['now'] = msg.clock.sec + msg.clock.nanosec * 1e-9

    def on_distance(msg):
        if state['goal'] is None and msg.data < goal_tol:
            state['goal'] = state['now']

    subs = [node.create_subscription(Clock, '/clock', on_clock, 10),
            node.create_subscription(Float64, '/flock/goal_distance', on_distance, 10)]
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.5)
        if state['now'] >= duration:
            break
        if state['goal'] is not None and state['now'] >= state['goal'] + settle:
            break
    for sub in subs:
        node.destroy_subscription(sub)
    return state['now'], state['goal'], time.monotonic() - start


def run_case(node, name, config, overrides, args):
    case_dir = os.path.join(RESULTS_DIR, name)
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)  # results of a previous run of this case
    os.makedirs(case_dir)
    params_file, params = write_params(case_dir, config, overrides)
    bag_dir = os.path.join(case_dir, 'bag')

    # Free the shared memory left by DDS participants that did not exit cleanly (see run_experiments.py)
    subprocess.run(['sh', shutil.which('fastdds') or 'fastdds', 'shm', 'clean'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(os.path.join(case_dir, 'record.log'), 'w') as rec_log, \
            open(os.path.join(case_dir, 'launch.log'), 'w') as launch_log:
        recorder = subprocess.Popen(
            ['ros2', 'bag', 'record', '-s', 'mcap', '--storage-preset-profile', 'zstd_fast',
             '--use-sim-time', '-p', '50',
             '-o', bag_dir, '-e', TOPICS_REGEX],
            stdout=rec_log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2.0)  # let the recorder start before the robots do
        launch = subprocess.Popen(
            ['ros2', 'launch', 'lab1_pkg', 'flock_navigation.launch.py',
             f'params_file:={params_file}', 'start_rviz:=false', 'start_plotjuggler:=false'],
            stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)

        reached, goal, wall = wait_for_goal(node, args.duration, args.settle, float(params['goal_tol']),
                                            timeout=3 * args.duration + 60)
        stop(recorder, 'recorder')
        stop(launch, 'launch')

    status = f'goal reached at {goal:.1f} s' if goal is not None else 'GOAL NOT REACHED'
    print(f'{name}: {status}, sim time {reached:.1f} s in {wall:.1f} s wall '
          f'(real time factor {reached / max(wall, 1e-6):.2f})', flush=True)


def parse_overrides(items):
    '''key=value strings -> dict, values typed by yaml (1.0 float, true bool, text string)'''
    overrides = {}
    for item in items:
        if '=' not in item:
            raise argparse.ArgumentTypeError(f'--set expects key=value, got {item}')
        key, value = item.split('=', 1)
        overrides[key] = yaml.safe_load(value)
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', nargs='*', help='run only these cases: number (nav2) or full name')
    parser.add_argument('--duration', type=float, default=200.0, help='simulated seconds, at most')
    parser.add_argument('--settle', type=float, default=10.0,
                        help='simulated seconds recorded after the goal is reached')
    parser.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE',
                        help='override parameters of every selected case (dots for nested keys)')
    parser.add_argument('--tag', default='', help='suffix of the results folder, to keep tuning runs apart')
    args = parser.parse_args()

    selected = {name: case for name, case in CASES.items()
                if not args.cases or name in args.cases or name.split('_')[0] in args.cases}
    unknown = [c for c in args.cases if not any(c in (n, n.split('_')[0]) for n in CASES)]
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")
    extra = parse_overrides(args.set)

    rclpy.init()
    node = rclpy.create_node('navigation_experiment_runner')
    try:
        for name, (config, overrides) in selected.items():
            run_name = f'{name}_{args.tag}' if args.tag else name
            run_case(node, run_name, config, {**overrides, **extra}, args)
            time.sleep(3.0)  # let every process of the case exit before the next one starts
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
