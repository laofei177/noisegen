from setuptools import setup, find_packages


setup(name='noisegen',
      version='1.0.0',
      description='A package to generate Gaussian distributed noise according to a user specified power spectral density.',
      packages=find_packages(),
      install_requires=[
            'matplotlib',
            'numpy',
            'pandas',
            'tqdm']
      )