import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'maze_cleanup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'maps'), glob(os.path.join('maps', '*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nikolasimon',
    maintainer_email='nikolasimon@uchicago.edu',
    description='Super Slam Dunk: Maze cleanup robot with particle filter localization and A* path planning',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'maze-cleanup = maze_cleanup.maze_cleanup:main',
            'particle-filter = maze_cleanup.particle_filter:main',
        ],
    },
)
