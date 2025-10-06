from setuptools import setup, find_packages
from pathlib import Path

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="fil3d",
    version="0.1.0",
    description="Detect and analyze coherent 3D filamentary structures in data cubes.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Doyeon Avery Kim & Larry Li",
    author_email="dakim@stsci.edu",
    url="https://github.com/a-dykim/fil3d",
    license="MIT",  # or "Apache-2.0"
    packages=find_packages(include=["fil3d", "fil3d.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
    ],
    extras_require={
        "cli": ["astropy>=5.0", "fil-finder>=1.7"],
        "all": ["astropy>=5.0", "fil-finder>=1.7"],
    },
    entry_points={
        "console_scripts": [
            "fil3d-find-trees = fil3d.cli.find_trees:main",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords=["3Dfilament", "image-processing", "PPV"],
    project_urls={
        "Source": "https://github.com/yourname/fil3d",
        "Tracker": "https://github.com/yourname/fil3d/issues",
    },
)
