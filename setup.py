from setuptools import setup, find_packages

def read_long_description():
    """Read the long description from README.md."""
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="fireant",
    version="1.0.0",
    author="Project FireAnt Team",
    author_email="team@fireant.dev",
    description="Production-ready agent orchestration framework with enterprise-grade features",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/sarankumar1325/FireAnt",
    project_urls={
        "Bug Reports": "https://github.com/sarankumar1325/FireAnt/issues",
        "Source": "https://github.com/sarankumar1325/FireAnt",
        "Documentation": "https://github.com/sarankumar1325/FireAnt#readme",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Framework :: AsyncIO",
    ],
    keywords="agent orchestration framework async monitoring persistence testing",
    install_requires=[
        # Core dependencies - minimal for basic functionality
        "typing-extensions>=4.0.0; python_version<'3.9'",
    ],
    extras_require={
        # Async support
        "async": [
            "aiofiles>=0.8.0",
        ],
        # Persistence with database support
        "persistence": [
            "aiofiles>=0.8.0",
            "sqlalchemy>=1.4.0",
        ],
        # Configuration with YAML support
        "config": [
            "pyyaml>=6.0",
        ],
        # Monitoring with advanced metrics
        "monitoring": [
            "psutil>=5.8.0",
            "prometheus-client>=0.14.0",
        ],
        # Testing utilities
        "testing": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
        ],
        # All optional dependencies
        "all": [
            "aiofiles>=0.8.0",
            "sqlalchemy>=1.4.0",
            "pyyaml>=6.0",
            "psutil>=5.8.0",
            "prometheus-client>=0.14.0",
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    keywords=[
        "agent", "orchestration", "workflow", "automation", "async",
        "monitoring", "persistence", "testing", "configuration"
    ],
    project_urls={
        "Documentation": "https://github.com/yourusername/fireant#readme",
        "Source": "https://github.com/yourusername/fireant",
        "Tracker": "https://github.com/yourusername/fireant/issues",
    },
    include_package_data=True,
    package_data={
        "fireant": ["py.typed"],
    },
)
