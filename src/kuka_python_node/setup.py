from setuptools import find_packages, setup

package_name = 'kuka_python_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),  # do not exclude test unless needed
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ram-micro',
    maintainer_email='ram-micro@ram-micro.nl',
    description='Python interface node for controlling KUKA robot.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kuka_commander = kuka_python_node.kuka_commander:main',
            'your_other_script = kuka_python_node.kuka_node:main',
        ],
    },
)
