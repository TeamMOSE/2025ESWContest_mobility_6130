from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament 인덱스 등록
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package.xml 설치
        ('share/' + package_name, ['package.xml']),
        # launch 파일들 설치
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daehyeon',
    maintainer_email='dh08080@khu.ac.kr',
    description='ROS2 nodes for rear camera publishing and YOLO-based firetruck detection',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = camera_pkg.camera_node:main',
            'yolo_firetruck_node = camera_pkg.yolo_firetruck_node:main',
            'yolo_trafficlight_node = camera_pkg.yolo_trafficlight_node:main',
        ],
    },
)