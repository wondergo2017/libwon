from setuptools import setup, find_packages

with open("readme.md", 'r') as fh:
    long_description = fh.read()
    
__version__ = '0.0.2'
URL = "https://github.com/wondergo2017/libwon/tree/main"
install_requires = [
    "matplotlib",
    "pandas",
    "requests"
]

setup(
    name='libwon',
    version=__version__,
    author='wondergo',
    author_email='wondergo2017@gmail.com',
    description='A simple lib to automatically manage the parallelled experiments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=URL,
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
