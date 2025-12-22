from setuptools import setup, find_packages

setup(
    name="mask2former",
    version="0.6",
    packages=find_packages(),
    install_requires=[],
    python_requires='>=3.10, <3.13', 
    description="Implementazione di Mask2Former basata su Detectron2."
)