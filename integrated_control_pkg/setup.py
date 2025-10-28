from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'integrated_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'pyserial',
    ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Integrated control package for emergency vehicle detection and lane management',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'integrated_control_node = integrated_control_pkg.integrated_control_node:main',
        ],
    },
)