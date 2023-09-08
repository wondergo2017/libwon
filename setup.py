from setuptools import setup, find_packages

__version__ = '0.0.1'
URL = None
install_requires = [
    "matplotlib",
    "pandas",
    "requests"
]

setup(
    name='libwon',
    version=__version__,
    description='libwon',
    author='wondergo',
    url=URL,
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
