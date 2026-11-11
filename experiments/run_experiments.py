#!/usr/bin/env python3
"""Run the Reynolds flocking cases in Stage and record one rosbag per case.

Run inside the container, with the workspace sourced:
    python3 run_experiments.py                 # all cases
    python3 run_experiments.py case2 case6     # only these cases (number or full name)
    python3 run_experiments.py --duration 20   # shorter runs, in simulated seconds

Each case gets a folder in results/ with its params file, the launch log and the bag.
Stage runs with its window open: only then does it keep simulated time in step with real time.
"""
import argparse
import copy
import os
import shutil
import signal
import subprocess
import time

# Keep the runs on their own DDS domain, away from any other session on this machine
os.environ['ROS_DOMAIN_ID'] = '42'
os.environ['ROS_LOCALHOST_ONLY'] = '1'

import rclpy  # noqa: E402  (must come after the environment is set)
import yaml  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_PARAMS = os.path.join(HERE, '..', 'lab1_pkg', 'config', 'reynolds_params.yaml')
RESULTS_DIR = os.path.join(HERE, 'results')
TOPICS_REGEX = r'^/clock$|^/robot_[0-9]+/(odom|cmd_vel|reynolds/.*)$'

# Values shared by every case; each case below changes only what it lists
BASELINE = {
    'number_robots': 10, 'ca': 0.5, 'cc': 0.1, 'cs': 0.1,
    'max_range': 20.0, 'fov': 360.0, 'f': 10.0,
    # Pinned so that editing the package yaml does not change reruns. 1.0 m/s^2 is above every
    # acceleration the cases use (largest: 0.86 in case 11), so the limit never acts by default.
    'brake_acc': 0.1, 'max_acc': 1.0,
    'map_name': 'empty_10x10', 'distribution': 'circle',
    'circle': {'center_x': 0.0, 'center_y': 0.0, 'radius': 2.0},
}
CASES = {
    'case1_baseline': {},
    'case2_weak_alignment': {'ca': 0.1},
    'case3_strong_separation': {'cs': 0.4},
    'case4_small_range': {'max_range': 1.5},
    'case5_small_range_strong_separation': {'max_range': 1.5, 'cs': 0.4},
    'case6_20_robots': {'number_robots': 20},
    # One rule at a time: the other two weights are zero
    'case7_pure_alignment': {'cc': 0.0, 'cs': 0.0},
    'case8_pure_separation': {'ca': 0.0, 'cc': 0.0},
    'case9_pure_cohesion': {'ca': 0.0, 'cs': 0.0},
    # Weights
    'case10_strong_alignment': {'ca': 2.0},
    'case11_strong_cohesion': {'cc': 0.4},
    # Number of agents
    'case12_4_robots': {'number_robots': 4},
    'case13_30_robots': {'number_robots': 30},
    # Neighborhood radius: at the start each robot sees only its two ring neighbors (1.24 m away)
    'case14_ring_range': {'max_range': 1.3},
    # Field of view
    'case15_fov_180': {'fov': 180.0},
    'case16_fov_90': {'fov': 90.0},
    # Map size: the 3 x 3 m map needs a smaller start circle
    'case17_small_map': {'map_name': 'empty_3x3',
                         'circle': {'center_x': 0.0, 'center_y': 0.0, 'radius': 1.0}},
    'case18_small_map_strong_separation': {'map_name': 'empty_3x3', 'cs': 0.4,
                                           'circle': {'center_x': 0.0, 'center_y': 0.0,
                                                      'radius': 1.0}},
}


def write_params(case_dir, overrides):
    '''Write the params file of a case: the package params file with the baseline and case values'''
    with open(BASE_PARAMS, 'r') as f:
        params = yaml.safe_load(f)
    ros_params = params['/**']['ros__parameters']
    for key, value in {**BASELINE, **overrides}.items():
        ros_params[key] = copy.deepcopy(value)
    path = os.path.join(case_dir, 'params.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(params, f, sort_keys=False)
    return path


def wait_for_sim_time(node, target, timeout):
    '''Block until /clock reaches target seconds; return (sim time reached, wall time spent)'''
    latest = [0.0]
    sub = node.create_subscription(
        Clock, '/clock', lambda msg: latest.__setitem__(0, msg.clock.sec + msg.clock.nanosec * 1e-9), 10)
    start = time.monotonic()
    while latest[0] < target and time.monotonic() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_subscription(sub)
    return latest[0], time.monotonic() - start


def stop(process, name, timeout=20.0):
    '''Send Ctrl+C to the whole process group, then escalate if it does not exit'''
    if process.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        os.killpg(process.pid, sig)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            print(f'  {name} did not stop on {sig.name}, escalating')


def run_case(node, name, overrides, duration):
    case_dir = os.path.join(RESULTS_DIR, name)
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)  # results of a previous run of this script
    os.makedirs(case_dir)
    params_file = write_params(case_dir, overrides)
    bag_dir = os.path.join(case_dir, 'bag')

    # Free the shared memory left by DDS participants that did not exit cleanly. The container's
    # /dev/shm has only 64 MB; when it fills up, nodes of a large swarm start without a transport.
    # Only unused segments are removed, so other running ROS processes are not affected.
    # fastdds is a shell script without a shebang line, so it must be run through sh
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
            ['ros2', 'launch', 'lab1_pkg', 'reynolds_rules.launch.py',
             f'params_file:={params_file}', 'start_rviz:=false', 'start_plotjuggler:=false'],
            stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)

        reached, wall = wait_for_sim_time(node, duration, timeout=3 * duration + 60)
        stop(recorder, 'recorder')
        stop(launch, 'launch')

    status = 'ok' if reached >= duration else 'INCOMPLETE'
    print(f'{name}: {status}, sim time {reached:.1f} s in {wall:.1f} s wall '
          f'(real time factor {reached / max(wall, 1e-6):.2f})', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', nargs='*', help='run only these cases: number (case2) or full name')
    parser.add_argument('--duration', type=float, default=100.0, help='simulated seconds per case')
    args = parser.parse_args()

    # Match the case number exactly, so case1 does not also select case10 to case18
    selected = {name: values for name, values in CASES.items()
                if not args.cases or name in args.cases or name.split('_')[0] in args.cases}
    unknown = [c for c in args.cases if not any(c in (n, n.split('_')[0]) for n in CASES)]
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")
    rclpy.init()
    node = rclpy.create_node('experiment_runner')
    try:
        for name, overrides in selected.items():
            run_case(node, name, overrides, args.duration)
            time.sleep(3.0)  # let every process of the case exit before the next one starts
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
