from setuptools import setup, find_packages

setup(
    name="dinov2-shotcut",
    version="0.1.0",
    description="DINOv2 shot segmentation & cutting tool",
    author="DAHE1998",
    url="https://github.com/DAHE1998/dinov2-shotcut",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24.0"],
)
