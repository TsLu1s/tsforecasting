import re
from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Single source of truth for the version: src/tsforecasting/__init__.py
version = re.search(
    r'^__version__ = "([^"]+)"',
    (this_directory / "src" / "tsforecasting" / "__init__.py").read_text(encoding="utf-8"),
    re.MULTILINE,
).group(1)

install_requires = [
    line.strip()
    for line in (this_directory / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setuptools.setup(
    name="tsforecasting",
    version=version,
    description="TSForecasting is an automated time series forecasting framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TsLu1s/TSForecasting",
    author="Luís Fernando da Silva Santos",
    author_email="luisf_ssantos@hotmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Customer Service",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Telecommunications Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    keywords=[
        "data science",
        "machine learning",
        "time series forecasting",
        "automated time series",
        "multivariate time series",
        "univariate time series",
        "automated machine learning",
        "automl",
    ],
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
            "build>=1.0",
            "twine>=4.0",
        ],
    },
    project_urls={
        "Source": "https://github.com/TsLu1s/TSForecasting",
        "Issues": "https://github.com/TsLu1s/TSForecasting/issues",
        "Changelog": "https://github.com/TsLu1s/TSForecasting/blob/main/CHANGELOG.md",
    },
)
