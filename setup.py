from distutils.core import setup
setup(
  name = 'multiple-recessives',
  packages = ['multiple-recessives'], # this must be the same as the name above
  version = '0.1',
  description = 'Programs for simulation of strategies for managing multiple recessives in a dairy cattle population. .',
  author = 'John B. Cole',
  author_email = 'john.cole@ars.usda.gov',
  url = 'https://github.com/wintermind/multiple-recessives',
  download_url = 'https://github.com/wintermind/multiple-recessives/tarball/0.1',
  keywords = ['simulation', 'dairy cattle', 'mating strategies', 'recessive disorders'],
  classifiers = ['Development Status :: 3 - Alpha', 'Intended Audience :: Science/Research',
	  'License :: Public Domain', 'Programming Language :: Python :: 2',
	  'Topic :: Scientific/Engineering'],
)
