import io
import os
from setuptools import setup, find_packages


def readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.rst')
    with io.open(readme_path) as f:
        return f.read()


setup(
    name='multimodal',
    version='0.0.1',
    description='A set of tools and experimental scripts used to achieve multimodal learning with nonnegative matrix factorization (NMF).',
    long_description=readme(),
    url='https://github.com/omangin/multimodal',
    author='Olivier Mangin',
    author_email='olivier.mangin@yale.edu',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'scipy', 'matplotlib', 'librosa'],
)
