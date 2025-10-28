from setuptools import setup

package_name = 'haptic_steering_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/haptic_steering_pkg']),
        ('share/haptic_steering_pkg', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daehyeon',
    maintainer_email='dh08080@gmail.com',
    description='Haptic steering decision node for emergency vehicle yielding.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'haptic_steering_node = haptic_steering_pkg.haptic_steering_node:main',
        ],
    },
)