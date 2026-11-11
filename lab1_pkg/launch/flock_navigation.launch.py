'''
Flock navigation: everything from reynolds_rules.launch.py (Stage, map_server, Reynolds
controllers, PlotJuggler) plus the flock planner and controller, and RViz with the flock path.

    ros2 launch lab1_pkg flock_navigation.launch.py
    ros2 launch lab1_pkg flock_navigation.launch.py start_plotjuggler:=false

The initial goal is goal_x / goal_y in the params file. '2D Goal Pose' in RViz sends a new one.
'''
import copy
import importlib.util
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# RViz displays for the flock navigation topics (latched topics need Transient Local durability)
LATCHED_TOPIC = {'Depth': 1, 'Durability Policy': 'Transient Local', 'History Policy': 'Keep Last',
                 'Reliability Policy': 'Reliable'}
FLOCK_DISPLAYS = [
    {'Class': 'rviz_default_plugins/Map', 'Name': 'FlockInflatedMap', 'Enabled': False,
     'Alpha': 0.3, 'Color Scheme': 'map', 'Draw Behind': True,
     'Topic': {**LATCHED_TOPIC, 'Value': '/flock/map_inflated'}, 'Value': False},
    {'Class': 'rviz_default_plugins/Path', 'Name': 'FlockPath', 'Enabled': True,
     'Color': '0; 255; 0', 'Line Style': 'Billboards', 'Line Width': 0.03, 'Pose Style': 'None',
     'Topic': {**LATCHED_TOPIC, 'Value': '/flock/path'}, 'Value': True},
    {'Class': 'rviz_default_plugins/PointStamped', 'Name': 'FlockCentroid', 'Enabled': True,
     'Color': '255; 0; 0', 'Radius': 0.08, 'History Length': 1,
     'Topic': {'Depth': 5, 'Durability Policy': 'Volatile', 'History Policy': 'Keep Last',
               'Reliability Policy': 'Reliable', 'Value': '/flock/centroid'}, 'Value': True},
]


def reynolds_launch_file():
    return os.path.join(get_package_share_directory('lab1_pkg'), 'launch', 'reynolds_rules.launch.py')


def write_flock_rviz_config(number_robots):
    '''The reynolds_rules rviz config (one RobotModel per robot) plus the flock displays'''
    # Load reynolds_rules.launch.py as a module to reuse its rviz config generation
    spec = importlib.util.spec_from_file_location('reynolds_rules_launch', reynolds_launch_file())
    reynolds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reynolds)

    template_file = os.path.join(get_package_share_directory('sphero_stage'), 'launch', 'sphero_sim.rviz')
    with open(reynolds.write_rviz_config(template_file, number_robots), 'r') as f:
        rviz = yaml.safe_load(f)
    rviz['Visualization Manager']['Displays'].extend(copy.deepcopy(FLOCK_DISPLAYS))

    output_file = os.path.join(reynolds.OUTPUT_DIR, f'flock_navigation_{number_robots}.rviz')
    with open(output_file, 'w') as f:
        yaml.safe_dump(rviz, f, sort_keys=False)
    return output_file


def launch_setup(context, *args, **kwargs):
    '''Flock planner + controller, and RViz with the flock displays'''
    params_file = LaunchConfiguration('params_file').perform(context)
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']
    number_robots = int(params['number_robots'])
    map_name = params['map_name']
    map_yaml = os.path.join(get_package_share_directory('sphero_stage'), 'config', 'maps', map_name, f'{map_name}.yaml')

    actions = [
        LogInfo(msg=f"Flock navigation on map '{map_name}', "
                    f"initial goal ({params.get('goal_x', 0.0)}, {params.get('goal_y', 0.0)})"),
        Node(
            package='lab1_pkg', executable='flock_planning_node', name='flock_planning_node',
            parameters=[params_file, {'map_yaml': map_yaml}],
            remappings=[('map', 'flock/map')],  # /map is already published by the map_server
            output='screen'),
        Node(
            package='lab1_pkg', executable='flock_controller_node', name='flock_controller_node',
            parameters=[params_file],
            output='screen'),
    ]

    # RViz is started here instead of in reynolds_rules.launch.py, to add the flock displays
    if (LaunchConfiguration('start_sim').perform(context) == 'true'
            and LaunchConfiguration('start_rviz').perform(context) == 'true'):
        urdf_file = os.path.join(
            get_package_share_directory('sphero_description'), 'urdf', 'simple_ball.urdf')
        with open(urdf_file, 'r') as f:
            robot_description = f.read()
        actions += [
            Node(
                package='robot_state_publisher', executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
                output='screen'),
            Node(
                package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', write_flock_rviz_config(number_robots)],
                parameters=[{'use_sim_time': True}],
                output='screen'),
        ]
    return actions


def generate_launch_description():
    share = get_package_share_directory('lab1_pkg')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(share, 'config', 'reynolds_params.yaml'),
            description='YAML file with the Reynolds, flock navigation and simulation parameters'),
        DeclareLaunchArgument(
            'start_sim', default_value='true',
            description='Also launch the Stage simulation (map from params_file)'),
        DeclareLaunchArgument(
            'start_rviz', default_value='true',
            description='Launch RViz with the flock path and centroid'),

        # Everything from reynolds_rules.launch.py except RViz, which is started below.
        # Scoped, so start_rviz:=false only applies inside the included file.
        GroupAction(scoped=True, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(reynolds_launch_file()),
                launch_arguments={'start_rviz': 'false'}.items()),
        ]),

        OpaqueFunction(function=launch_setup),
    ])
