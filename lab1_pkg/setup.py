import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'lab1_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'plotjuggler'), glob('plotjuggler/*.xml')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='david.villanueva.guzman@gmail.com',
    description='Reynolds flocking rules for a swarm of robots in Stage',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'reynolds_node = lab1_pkg.reynolds_node:main',
            'flock_planning_node = lab1_pkg.flock_planning_node:main',
            'flock_controller_node = lab1_pkg.flock_controller_node:main',
        ],
    },
)
