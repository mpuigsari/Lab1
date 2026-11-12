import copy
import math
import os
import xml.etree.ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROBOT_LINE = 'robot ( pose [ {x:.3f} {y:.3f} 0 0.000 ] name "robot_{k}" color "blue")\n'

# The generated rviz config and plotjuggler layout are written here
OUTPUT_DIR = os.path.expanduser('~/.ros/lab1_pkg')

# One color per robot trajectory in the plotjuggler XY plot (cycled if there are more robots)
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# RViz display of flock_markers_node
MARKERS_DISPLAY = {
    'Class': 'rviz_default_plugins/MarkerArray', 'Name': 'FlockMarkers', 'Enabled': True, 'Namespaces': {},
    'Topic': {'Depth': 100, 'Durability Policy': 'Volatile', 'History Policy': 'Keep Last',
              'Reliability Policy': 'Reliable', 'Value': '/flock/markers'},
    'Value': True}
TOP_DOWN_VIEW = {
    'Class': 'rviz_default_plugins/TopDownOrtho', 'Name': 'Current View', 'Value': 'TopDownOrtho (rviz_default_plugins)',
    'Angle': 0.0, 'Scale': 70.0, 'X': 0.0, 'Y': 0.0, 'Invert Z Axis': False,
    'Near Clip Distance': 0.01, 'Target Frame': '<Fixed Frame>'}


def markers_node(number_robots):
    '''Robots colored by obstacle avoidance + trajectories, shown in RViz'''
    return Node(
        package='lab1_pkg', executable='flock_markers_node', name='flock_markers_node',
        parameters=[{'number_robots': number_robots, 'use_sim_time': True}],
        output='screen')


def distribute_circle(k, n, center_x=0, center_y=0, radius=1):
    '''Robots evenly distributed on a circle'''
    x = radius * math.cos(k / n * 2 * math.pi) + center_x
    y = radius * math.sin(k / n * 2 * math.pi) + center_y
    return x, y


def distribute_line(k, n, center_x=0, center_y=0, separation=0.5, direction='horizontal'):
    '''Robots in a straight line'''
    offset = (k - (n - 1) / 2) * separation
    if direction == 'horizontal':
        return center_x + offset, center_y
    return center_x, center_y + offset


def write_rviz_config(template_file, number_robots):
    '''Copy the sphero rviz config with one RobotModel display per robot, plus the flock markers
    (robots colored by obstacle avoidance and their trajectories), in a top-down view. Return its path'''
    with open(template_file, 'r') as f:
        rviz = yaml.safe_load(f)

    displays = rviz['Visualization Manager']['Displays']
    robot_models = [d for d in displays if d['Class'] == 'rviz_default_plugins/RobotModel']
    if not robot_models:
        raise RuntimeError(f'No RobotModel display found in {template_file}')

    # Replace the fixed robot displays with one copy per robot, at the same place in the list.
    # They are off by default: the flock markers draw the robots with the obstacle avoidance color
    index = displays.index(robot_models[0])
    displays[:] = [d for d in displays if d['Class'] != 'rviz_default_plugins/RobotModel']
    for i in range(number_robots):
        model = copy.deepcopy(robot_models[0])
        model['Name'] = f'RobotModel{i}'
        model['TF Prefix'] = f'/robot_{i}'
        model['Enabled'] = False
        displays.insert(index + i, model)

    # The robot frames clutter the trajectories: TF off by default (can be enabled in RViz)
    for d in displays:
        if d['Class'] == 'rviz_default_plugins/TF':
            d['Enabled'] = False
    displays.append(copy.deepcopy(MARKERS_DISPLAY))

    # Top-down view of the whole 10 x 10 m map, to look at the trajectories
    rviz['Visualization Manager']['Views']['Current'] = copy.deepcopy(TOP_DOWN_VIEW)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f'reynolds_{number_robots}.rviz')
    with open(output_file, 'w') as f:
        yaml.safe_dump(rviz, f, sort_keys=False)
    return output_file


def expand_foreach_robot(root, number_robots):
    '''Replace every element marked foreach_robot="true" by one copy per robot, with {i} replaced by
    the robot index and {color} by a palette color (e.g. one XY curve per robot trajectory)'''
    for parent in list(root.iter()):
        children = []
        for child in parent:
            if child.get('foreach_robot') != 'true':
                children.append(child)
                continue
            for i in range(number_robots):
                copy_ = copy.deepcopy(child)
                del copy_.attrib['foreach_robot']
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                for key, value in copy_.attrib.items():
                    copy_.set(key, value.replace('{i}', str(i)).replace('{color}', color))
                children.append(copy_)
        parent[:] = children


def write_plotjuggler_layout(template_file, robot, number_robots):
    '''Expand the plotjuggler layout template and return its path: {robot} is the monitored robot,
    foreach_robot elements and {i} topics are repeated for every robot'''
    with open(template_file, 'r') as f:
        root = ET.fromstring(f.read().replace('{robot}', str(robot)))
    expand_foreach_robot(root, number_robots)

    # Topics with {i} are subscribed for every robot. The monitored robot can also be one of the
    # overview robots: drop repeated topics
    topics = root.find('.//selected_topics')
    if topics is not None:
        expanded = []
        for topic in topics.get('value').split(';'):
            if '{i}' in topic:
                expanded += [topic.replace('{i}', str(i)) for i in range(number_robots)]
            else:
                expanded.append(topic)
        topics.set('value', ';'.join(dict.fromkeys(expanded)))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f'reynolds_layout_robot_{robot}.xml')
    ET.ElementTree(root).write(output_file, encoding='UTF-8', xml_declaration=True)
    return output_file


def simulation_nodes(params, number_robots, start_rviz):
    '''Create the world file from a predefined sphero_stage map and return the simulation nodes'''
    sphero_share = get_package_share_directory('sphero_stage')
    map_name = params['map_name']
    template_file = os.path.join(sphero_share, 'config', 'world_templates', f'{map_name}.temp')
    map_yaml = os.path.join(sphero_share, 'config', 'maps', map_name, f'{map_name}.yaml')
    if not os.path.isfile(template_file) or not os.path.isfile(map_yaml):
        available = sorted(f[:-len('.temp')] for f in os.listdir(os.path.dirname(template_file)))
        raise RuntimeError(f"Unknown map '{map_name}'. Available maps: {', '.join(available)}")

    # World file = map template + robots. Written next to the sphero_stage worlds so the
    # relative bitmap path and the include in the template resolve.
    distribution = params['distribution']
    distribute = {'circle': distribute_circle, 'line': distribute_line}[distribution]
    world_file = os.path.join(sphero_share, 'config', 'worlds', f'{map_name}_{number_robots}.world')
    with open(world_file, 'w') as output, open(template_file, 'r') as template:
        output.write(template.read())
        for k in range(number_robots):
            x, y = distribute(k, number_robots, **params[distribution])
            output.write(ROBOT_LINE.format(x=x, y=y, k=k))

    nodes = [
        LogInfo(msg=f"Map '{map_name}' with {number_robots} robots ({distribution}): {world_file}"),
        Node(
            package='nav2_map_server', executable='map_server', name='map_server',
            parameters=[{'yaml_filename': map_yaml, 'use_sim_time': True}],
            output='screen'),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_map_server',
            parameters=[{'node_names': ['map_server'], 'use_sim_time': True, 'autostart': True}],
            output='screen'),
        Node(
            package='stage_ros2', executable='stage_ros2', name='simulator',
            parameters=[{'world_file': world_file, 'enforce_prefixes': False,
                         'one_tf_tree': True}],
            output='screen'),
    ]
    for i in range(number_robots):
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'static_tf_publisher_robot_{i}',
            arguments=['--frame-id', 'map', '--child-frame-id', f'robot_{i}/odom']))

    if start_rviz:
        urdf_file = os.path.join(
            get_package_share_directory('sphero_description'), 'urdf', 'simple_ball.urdf')
        with open(urdf_file, 'r') as f:
            robot_description = f.read()
        rviz_config = write_rviz_config(
            os.path.join(sphero_share, 'launch', 'sphero_sim.rviz'), number_robots)
        nodes += [
            Node(
                package='robot_state_publisher', executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
                output='screen'),
            Node(
                package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', rviz_config],
                parameters=[{'use_sim_time': True}],
                output='screen'),
            markers_node(number_robots),
        ]
    return nodes


def launch_setup(context, *args, **kwargs):
    '''Read the params file, start the simulation and spawn one controller per robot'''
    params_file = LaunchConfiguration('params_file').perform(context)

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']
    number_robots = int(params['number_robots'])

    actions = [LogInfo(msg=f'Launching {number_robots} Reynolds controllers from {params_file}')]

    if LaunchConfiguration('start_sim').perform(context) == 'true':
        start_rviz = LaunchConfiguration('start_rviz').perform(context) == 'true'
        actions += simulation_nodes(params, number_robots, start_rviz)

    if LaunchConfiguration('start_plotjuggler').perform(context) == 'true':
        robot = int(LaunchConfiguration('monitor_robot').perform(context))
        if not 0 <= robot < number_robots:
            raise RuntimeError(
                f'monitor_robot={robot}, but only robots 0 to {number_robots - 1} exist')
        layout = write_plotjuggler_layout(
            LaunchConfiguration('plotjuggler_layout').perform(context), robot, number_robots)
        actions.append(Node(
            package='plotjuggler', executable='plotjuggler', name='plotjuggler',
            arguments=['--layout', layout],
            output='screen'))

    for i in range(number_robots):
        actions.append(Node(
            package='lab1_pkg',
            executable='reynolds_node',
            name=f'reynolds_node_{i}',  # unique name, no namespace: topics are already robot_{i}/...
            parameters=[params_file, {'robot_id': i}],
            output='screen',
        ))

    return actions


def generate_launch_description():
    share = get_package_share_directory('lab1_pkg')
    default_params = os.path.join(share, 'config', 'reynolds_params.yaml')
    default_layout = os.path.join(share, 'plotjuggler', 'reynolds_layout.xml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='YAML file with the Reynolds controller and simulation parameters'),
        DeclareLaunchArgument(
            'start_sim', default_value='true',
            description='Also launch the Stage simulation (map from params_file)'),
        DeclareLaunchArgument(
            'start_rviz', default_value='true',
            description='Launch RViz with the sphero_stage config'),
        DeclareLaunchArgument(
            'start_plotjuggler', default_value='true',
            description='Launch PlotJuggler with the velocity/acceleration layout'),
        DeclareLaunchArgument(
            'plotjuggler_layout', default_value=default_layout,
            description='PlotJuggler layout file, {robot} is replaced by monitor_robot, '
                        'foreach_robot elements and {i} topics are repeated for every robot'),
        DeclareLaunchArgument(
            'monitor_robot', default_value='0',
            description='Robot shown in the single robot PlotJuggler tabs'),

        OpaqueFunction(function=launch_setup),
    ])
