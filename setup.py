from setuptools import find_packages, setup

package_name = 'omy_impedance_control'

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
    maintainer='root',
    maintainer_email='guseo0328@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cartesian_impedance_node = omy_impedance_control.cartesian_impedance_node:main',
            'cartesian_impedance_real_node = '
            'omy_impedance_control.cartesian_impedance_real_node:main',
            'cartesian_impedance_real_teleop_node = '
            'omy_impedance_control.cartesian_impedance_real_teleop_node:main',
            'cartesian_impedance_position_real_node = '
            'omy_impedance_control.cartesian_impedance_position_real_node:main',
            'leader_tcp_target_node = '
            'omy_impedance_control.leader_tcp_target_node:main',
        ],
    },
)
