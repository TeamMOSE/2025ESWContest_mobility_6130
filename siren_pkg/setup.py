import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'siren_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.pth')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'matplotlib',
        'pyaudio',
        'scipy',
        'opencv-python',  # cv2
        'torch',
        'torchvision',
        'Pillow',         # PIL
    ],
    zip_safe=True,
    maintainer='mose',
    maintainer_email='mose@localhost',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fft_ros2_publisher = siren_pkg.fft_ros2_publisher:main',
            'siren_detector_node = siren_pkg.siren_detector_node:main',
            'direction_detector_node = siren_pkg.direction_detector_node:main',
            'siren_node = siren_pkg.siren_node:main',
            'image_viewer_node = siren_pkg.image_viewer_node:main'
        ],
    },
)



