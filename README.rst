====
Adaptation of brainreg for Get Your Brain Straight hackathon
====

A package to support the following contribution:
https://github.com/InsightSoftwareConsortium/GetYourBrainStraight/tree/main/HCK01_2022_Virtual/ReproducibleResource/IanaVasylieva

.. image:: https://img.shields.io/pypi/v/gybs.svg
        :target: https://pypi.python.org/pypi/gybs

.. image:: https://img.shields.io/travis/noisysky/gybs.svg
        :target: https://travis-ci.com/noisysky/gybs

.. image:: https://readthedocs.org/projects/gybs/badge/?version=latest
        :target: https://gybs.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


* Free software: BSD license

This package uses **brainreg** open-source image registration software: https://github.com/brainglobe/brainreg

Installation (standalone)
*************************
::

    conda create -y -n brainreg python=3.8
    conda activate brainreg
    git clone https://github.com/noisysky/GYBS_hackathon.git
    cd GYBS_hackathon
    pip install -r requirements.txt

Usage (standalone)
******************
::

    gybs -i /path/to/input_img.nii.gz -o /path/to/output_folder -v 10 10 10 --orientation sla

refer to **brainreg** documentation: https://docs.brainglobe.info/brainreg/tutorial


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
