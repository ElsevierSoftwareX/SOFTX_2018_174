#from distutils.core import setup
from setuptools import setup

setup(name='fieldAnimation',
    version='0.1',
    description='Animate 2D vector fields',
    author='Nicola Creati',
    url='http://www.inogs.it/it/users/nicola-creati',
    packages=['fieldAnimation'],
    package_data={
        'fieldAnimation': ['glsl/*', ],
    }
 )
