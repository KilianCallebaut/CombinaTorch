from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='CombinaTorch',
    version='0.1',
    packages=find_packages(),
    description='A Python framework built on top of PyTorch for building Multi-dataset, Multi-Task deep learning pipelines',
    author='Kilian Callebaut',
    author_email='kilian.callebaut@gmail.com',
    url='https://github.com/yourusername/mypackage',  # Optional
    install_requires=requirements,
)