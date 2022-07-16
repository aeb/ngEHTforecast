#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer
import glob

# script_dir = glob.glob('scripts/ngEHTforecast_*')
# script_files = []
# for sf in script_dir :
#     if ( sf[-1]!='~' ) :
#         script_files.append(sf)
        
setup(name='ngEHTforecast',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='ngEHT science forecasting tools',
      author='Avery E. Broderick',
      author_email='abroderick@perimeterinstitute.ca',
      url='https://github.com/aeb/ngEHTforecast',
      packages=find_packages(),
      install_requires=['numpy','scipy','matplotlib','ehtim'],
      # scripts=script_files
     )

