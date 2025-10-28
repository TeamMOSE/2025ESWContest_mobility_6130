from setuptools import setup

package_name = 'velocity_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/velocity_pkg']),
        ('share/velocity_pkg', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daehyeon',
    maintainer_email='dh08080@gmail.com',
    description='Velocity node that reads serial analog values and publishes velocity topic.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'velocity_node = velocity_pkg.velocity_node:main',
        ],
    },
)

