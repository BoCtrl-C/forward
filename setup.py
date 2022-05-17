from setuptools import find_packages, setup


setup(
    name='forward',
    version='dev',
    url='https://github.com/BoCtrl-C/forward',
    author='Tommaso Boccato',
    author_email='tommaso.boccato@uniroma2.it',
    packages=find_packages(),
    install_requires=[
        'networkx'
        'torch'
    ]
)