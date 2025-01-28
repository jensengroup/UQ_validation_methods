from setuptools import setup
import os

# requirements
lib_path = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(lib_path, 'requirements.txt')
with open(requirements_path) as f:
    install_requires = f.read().splitlines()


setup(
    name="UQ_validation_methods",
    description="Methods for evaluating uncertainty quantification metrics",
    url="https://github.com/jensengroup/UQ_validation_methods",
    py_modules=["UQtools"],
    install_requires=install_requires
)