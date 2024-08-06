from setuptools import find_packages, setup

setup(
    name="test_fake_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Rohin Bhasin",
    author_email="bhasin.rohin@gmail.com",
    description="A simple example package",
    python_requires=">=3.6",
)
