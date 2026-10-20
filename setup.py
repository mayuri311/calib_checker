from glob import glob

from setuptools import setup

package_name = "calib_checker"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mayuri Murugan",
    maintainer_email="mmurugan@andrew.cmu.edu",
    description="Live static-hold guidance and online view selection for cross-modal calibration.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "checker = calib_checker.ros_node:main",
            "calib_checker_replay = calib_checker.replay:main",
        ],
    },
)
