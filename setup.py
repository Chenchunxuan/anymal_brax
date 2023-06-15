import os
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstallCommand(install):
    # This is only run for "python setup.py install" (not for "pip install -e .")
    def run(self):
        print("--------------------------------")
        print("Installing anymal_brax ...")
        print("--------------------------------")
        install.run(self)
        print("--------------------------------")
        print("...done.")
        print("--------------------------------")


setup(name='anymal_brax',
      version='1.0',
      author='Julian Nubert (nubertj@ethz.ch)',
      package_dir={"": "src"},
      install_requires=[
      ],
      scripts=['bin/test_jax_gpu.py', 'bin/train_reacher_ppo.py', 'bin/train_reacher_apg.py'],
      license='BSD-3-Clause',
      description='ANYmal BRAX',
      cmdclass={'install': CustomInstallCommand, },
      )