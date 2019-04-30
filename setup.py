from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='crystalml',
      version='0.0.1',
      description='Integrated tool to measure the nucleation rate of protein crystals. ',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/hlgirard/CrystalML',
      author='Henri-Louis Girard',
      author_email='hl.girard@gmail.com',
      license='GPLv3',
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'tensorflow',
          'matplotlib',
          'numpy',
          'opencv-python',
          'pandas',
          'pillow',
          'plotly',
          'scikit-image',
          'setuptools',
          'scipy',
          'seaborn',
          'tensorboard',
          'tqdm'

      ],
      scripts=[
          'bin/crystalml',
      ],
      zip_safe=False,
      include_package_data=True)