from setuptools import setup, find_packages

setup(
    name="brainnet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'kagglehub>=0.1.0',
        'Pillow>=8.0.0',
        'tqdm>=4.65.0',
    ],
) 