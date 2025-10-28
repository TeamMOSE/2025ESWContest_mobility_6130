from setuptools import find_packages, setup

package_name = 'debug_pkg'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DaeHyeon Kim',
    maintainer_email='dh08080@khu.ac.kr',
    description='Debug visualizers for YOLO detections (firetruck overlay on camera)',
    license='Apache-2.0',
    # 최신 setuptools에서는 tests_require 경고가 떠서 제거
    entry_points={
        'console_scripts': [
            # 실행 엔트리 등록
            'firetruck_visualizer_node = debug_pkg.firetruck_visualizer_node:main',
        ],
    },
)