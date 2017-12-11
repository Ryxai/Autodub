try:
    from setuptools import setup
except:
    from distutils.core import setup

config = {
    "name": "autodub",
    "description": "A tool to dub the melody of an existing track with " +
                    "a user provided set of 'dub' tracks",
    "author": "Ryxai",
    "version": "0.0.1",
    "install_requires": ["numpy", "scipy", "vamp",
                                    "pydub", "librosa", "functools"],
    "packages": ["autodub"],
    "scripts": ["src/autodub.py"],
    "setup_requires": ["pytest_runner"],
    "tests_require":["pytest"]
}

setup(**config)