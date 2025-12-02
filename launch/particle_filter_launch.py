from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare a required launch argument called "namespace"
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        description='Namespace for the robot (required)'
    )

    # Use the argument as a LaunchConfiguration substitution
    namespace = LaunchConfiguration('namespace')

    return LaunchDescription([
        namespace_arg,
        Node(
            package='maze_cleanup',
            executable='particle-filter',
            namespace=namespace,
            remappings=[
                ('/tf', [namespace, '/tf']),
                ('/tf_static', [namespace, '/tf_static']),
            ],
            output='screen'
        ),
    ])

