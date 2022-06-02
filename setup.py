from setuptools import setup, find_packages


setup(name='SPLR',
      version='1.1.0',
      description='Soft pseudo-label shrinkage for unsupervised domain adaptive person re-ID',
      install_requires=[
          'numpy', 'torch==1.1.0', 'torchvision==0.2.2', 
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Domain Adaptation',
          'Person Re-identification',
          'Deep Learning',
      ])
