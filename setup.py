"""
项目设置文件
"""

from setuptools import setup, find_packages

setup(
    name='a_gcs_ws',
    version='2.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
)
