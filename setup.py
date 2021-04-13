from setuptools import setup, find_packages


with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="syke-pic",
    version="0.2.0",
    packages=find_packages(),
    description="Plankton image classification with neural networks",
    long_description=long_description,
    entry_points={
        "console_scripts": [
            "sykepic = sykepic.__main__:main",
        ],
    },
    author="Otso Velhonoja",
    author_email="otso.velhonoja@syke.fi",
    url="https://github.com/veot/syke-pic",
)
