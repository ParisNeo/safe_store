from setuptools import setup, find_packages

# Package metadata
NAME = 'safe_store'
DESCRIPTION = 'A library for safe document storage and vectorization.'
URL = 'https://github.com/ParisNeo/safe_store'
AUTHOR = 'ALOUI Saifeddine (ParisNeo)'
AUTHOR_EMAIL = 'aloui.seifeddine@email.com'
LICENSE = 'Apache 2.0'
VERSION = '0.1.2'  # Update with your desired version

# Read dependencies from requirements.txt
with open('requirements.txt', 'r') as req_file:
    INSTALL_REQUIRES = req_file.read().splitlines()

# Packages to include (find_packages() automatically discovers and includes sub-packages)
PACKAGES = find_packages()

# Define long description (usually from README.md)
with open('README.md', 'r') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

# Additional package data (e.g., data files)
PACKAGE_DATA = {}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
