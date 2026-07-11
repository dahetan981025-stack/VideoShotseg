from setuptools import setup, find_packages

setup(
    name="VideoShotseg",
    version="0.1.0",
    description="基于聚类分析的视频镜头分割工具",
    author="dahetan981025-stack",
    url="https://github.com/dahetan981025-stack/VideoShotseg",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24.0"],
)
