from pathlib import Path
from setuptools import find_packages, setup
import sys

sys.path.insert(0, str(Path(__file__).parent))
from local_notebooklm.version import __version__

setup(
    name="local-notebooklm",
    version=__version__,
    description="A Local-NotebookLM to convert PDFs into Audio.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Gökdeniz Gülmez",
    author_email="goekdenizguelmez@gmail.com",
    url="https://github.com/Goekdeniz-Guelmez/Local-NotebookLM",
    license="Apache-2.0",
    # Remove install_requires=requirements,
    packages=find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.12"
    ],
    include_package_data=True
)