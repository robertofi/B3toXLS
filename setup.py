from setuptools import setup, find_packages

# requirements = ["streamlit", "pandas", "numpy", 'datetime']

setup(
    name="B3toXLS",
    version="0.0.0",
    author="Roberto Fix Ventura",
    author_email="rob@ventur.as",
    description="Quant Strategies",
    long_description_content_type="text/markdown",
    url="https://github.com/robertofi/B3toXLS/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: MIT",
    ],
)

# to install:
# pip install git+"https://github.com/robertofi/B3toXLS/"