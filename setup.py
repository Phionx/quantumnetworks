"""
Setup for quantumnetworks
"""
import os

from setuptools import setup, find_namespace_packages

REQUIREMENTS = [
    "numpy",
    "scipy",
    "matplotlib>=3.3.0",
    "IPython",
    "networkx",
    "tqdm",
]
EXTRA_REQUIREMENTS = {"dev": ["jupyterlab>=3.1.0"]}

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "quantumnetworks", "VERSION.txt")
)

with open(version_path, "r") as fd:
    version_str = fd.read().rstrip()

setup(
    name="quantumnetworks",
    version=version_str,
    description="Quantum Networks",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Phionx/quantumnetworks",
    author="Shantanu Jha, Shoumik Chowdhury, Lamia Ateshian",
    author_email="shantanu.rajesh.jha@gmail.com",
    license="MIT",
    packages=find_namespace_packages(exclude=["tests*", "demos*"]),
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    keywords="quantum networks",
    python_requires=">=3.6",
    project_urls={
        "Bug Tracker": "https://github.com/Phionx/quantumnetworks/issues",
        "Documentation": "https://github.com/Phionx/quantumnetworks",
        "Source Code": "https://github.com/Phionx/quantumnetworks",
        "Demos": "https://github.com/Phionx/quantumnetworks/demos",
        "Tests": "https://github.com/Phionx/quantumnetworks/tests",
    },
    include_package_data=True,
)
