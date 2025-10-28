from setuptools import find_packages, setup

package_name = 'lane_detector_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cyjung',
    maintainer_email='cyjung@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lane_detector_node = lane_detector_pkg.lane_detector_node:main',
            'lane_state_switcher = lane_detector_pkg.lane_state_switcher:main',
        ],
    },
)
