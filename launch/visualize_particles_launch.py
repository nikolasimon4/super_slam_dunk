from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_maze_cleanup = get_package_share_directory('maze_cleanup')

    # --- Declare arguments ---
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace to apply to all topics'
    )

    map_arg = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution(
            [pkg_maze_cleanup, 'maps', 'maze_map.yaml']),
        description='Full path to the map YAML file to load'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # --- Launch configurations ---
    namespace = LaunchConfiguration('namespace')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # --- Group under namespace ---
    map_group = GroupAction([
        PushRosNamespace(namespace),

        # 1️ Map server node (lifecycle)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_yaml_file,
                'use_sim_time': use_sim_time
            }],
            respawn=True,
            respawn_delay=0.5,
        ),

        # 2️ Lifecycle manager to automatically activate it
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': ['map_server']
            }]
        )
    ])

    # --- 3️ RViz node (custom visualization) ---
    rviz_config = PathJoinSubstitution(
        [pkg_maze_cleanup, 'rviz', 'visualize_particles_and_map.rviz']
    )

    rviz_node = GroupAction([
        PushRosNamespace(namespace),

        # Delay slightly so map server is ready before RViz loads
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    output='screen',
                    arguments=['-d', rviz_config],
                    parameters=[{'use_sim_time': use_sim_time}],
                    remappings=[
                        # Map and TF are typically global; adjust as needed for namespaced use
                        ('tf', '/tf'),
                        ('tf_static', '/tf_static'),
                        ('map', '/map'),
                        ('map_metadata', '/map_metadata')
                    ],
                )
            ]
        )
    ])

    # --- Build the launch description ---
    ld = LaunchDescription()
    ld.add_action(namespace_arg)
    ld.add_action(map_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(map_group)
    ld.add_action(rviz_node)

    return ld

