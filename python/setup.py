from setuptools import setup
from setuptools import find_packages


setup(name='glrec',
      version='0.1.0',
      description='Convenient library for Google Landmark Challenge 2020',
      author='Chan Kha Vu',
      url='https://github.com/hav4ik/google-landmarks-2020',
      install_requires=['tensorflow>=2.2.0',
                        'colorlog>=4.1.0'],
      packages=find_packages())
