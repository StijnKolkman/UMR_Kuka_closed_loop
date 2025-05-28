from setuptools import find_packages, setup
from glob import glob

package_name = "kukavarproxy_ros"
submodule = "kukapythonvarproxy"
setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(include=[package_name, submodule]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jelle",
    maintainer_email="j.p.idzenga@student.utwente.nl",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "kuka_control = " + package_name + ".kuka_control:main",
            "action_pub = kukavarproxy_ros.kuka_action:main",
            "csv_path = kukavarproxy_ros.csv_kuka:main",
            "magnet_control = kukavarproxy_ros.control_policy:main",
            'kuka_commander = kukavarproxy_ros.kuka_commander:main',
        ],
    },
)
