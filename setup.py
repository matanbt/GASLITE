from setuptools import find_packages, setup

setup(
    name="gaslite",
    version="0.0.1",
    description="Poison a retrieval corpus w/ GASLITE; Evaluate post-poisoning retrieval",
    long_description=open("README.md").read(),
    author="Matan Ben-Tov",
    author_email="matanbentov@mail.tau.ac.il",
    packages=find_packages(),
    license="LICENSE",
    install_requires=open("requirements.txt").readlines(),
)
