from setuptools import find_packages, setup
import os

package_name = "maxon"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("lib", package_name), ["include/EposCmd64.dll"]),
        (os.path.join("lib", package_name), ["include/libEposCmd.so.6.8.1.0"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jelle",
    maintainer_email="j.p.idzenga@student.utwente.nl",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "maxon_node = maxon.maxon_node:main",
        ],
    },
)
