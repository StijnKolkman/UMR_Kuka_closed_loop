from setuptools import setup, find_packages

setup(
   name="KukaPythonProxy",
   author="Jelle Idzenga, Ewout Ligtenberg",
   description="Private Python library for Kuka control in pyton",
   packages=find_packages(),
   include_package_data=True,
   classifiers=[
       "Environment :: Web Environment",
       "Intended Audience :: Developers",
       "Operating System :: OS Independent"
       "Programming Language :: Python",
       "Programming Language :: Python :: 3.6",
       "Topic :: Internet :: WWW/HTTP",
       "Topic :: Internet :: WWW/HTTP :: Dynamic Content"
   ],
   python_requires='>=3.0',
   setup_requires=['setuptools-git-versioning'],
   version_config={
       "dirty_template": "{tag}",
   }
)