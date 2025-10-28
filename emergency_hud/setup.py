from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'emergency_hud'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyomin',
    maintainer_email='rhaygls1008@gmail.com',
    description='Emergency vehicle HUD display system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emergency_hud_node = emergency_hud.emergency_hud_node:main',
        ],
    },
)
