from setuptools import setup

setup(name='fqs',
      version='0.1.0',
      packages=["fqs"],
      package_dir={"": "src"},
      description="A differentiable fast quartic solver written in TensorFlow.")
