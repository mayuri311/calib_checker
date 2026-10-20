"""ros2 launch calib_checker checker.launch.py [session:=<yaml>] [viz_host:=192.168.50.202] [headless:=false]"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        "session": "",
        "headless": "false",
        "viz_host": "192.168.50.202",
        "grpc_port": "9876",
        "web_port": "9090",
        "ui_port": "8080",
        "output_dir": "",
    }
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
        + [Node(package="calib_checker", executable="checker", name="calib_checker", output="screen",
                parameters=[{k: LaunchConfiguration(k) for k in args}])]
    )
