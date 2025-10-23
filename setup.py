from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="nert-safety",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    author="Anonymous",
    description="NERT: Neurosymbolic Embodied Reasoning for Task Safety",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
