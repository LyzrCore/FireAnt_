from setuptools import setup, find_packages

def read_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="fireant",
    version="0.1.0",
    author="Project FireAnt Team",
    author_email="team@fireant.dev",
    description="Minimal agent orchestration framework (85 lines core)",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
