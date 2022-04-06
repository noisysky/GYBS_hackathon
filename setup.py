#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Iana Vasylieva",
    author_email='iana.vasylieva@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Get Your Brain Straight hackathon",
    entry_points={
        'console_scripts': [
            'gybs=gybs.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='GYBS_hackathon',
    name='GYBS_hackathon',
    packages=find_packages(include=['gybs', 'gybs.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/noisysky/gybs',
    version='0.1.0',
    zip_safe=False,
)
