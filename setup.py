from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='crystalml',
      version='0.0.4',
      description='Integrated tool to measure the nucleation rate of protein crystals. ',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/hlgirard/CrystalML',
      author='Henri-Louis Girard',
      author_email='hl.girard@gmail.com',
      license='GPLv3',
      packages=find_packages(exclude=["tests.*", "tests"]),
      install_requires=[
          'click',
          'bleach>=2.1.0',
          'docutils>=0.13.1',
          'Pygments',
          'tensorflow',
          'matplotlib',
          'numpy',
          'opencv-python',
          'pkginfo>=1.4.2',
          'pandas',
          'joblib',
          'pillow',
          'plotly',
          'scikit-image',
          'setuptools',
          'scipy',
          'seaborn',
          'tensorboard',
          'tqdm',
      ],
      entry_points={
          'console_scripts': [
              'crystalml = src.cli:cli',
          ],
      },
      zip_safe=False,
      include_package_data=True)