"""
Package setup configuration for the Anime Chatbot.

This module defines the package metadata, dependencies, and installation
configuration for the anime_chatbot package using setuptools. It specifies
required packages, test dependencies, and package discovery settings.
"""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'google-genai==1.28.0',
]

TEST_EXTRAS = [
    'pytest>=8.0.0',
]

setup(
    name='anime_chatbot',
    version='0.1',
    author='WaifuAI',
    author_email='waifuai@users.noreply.github.com',
    url='https://github.com/waifuai/anime-subtitle-chatbot-trax',
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        'test': TEST_EXTRAS,
    },
    packages=find_packages(),
    include_package_data=True,
    description='Anime Chatbot Problem',
    requires=[]  # Note: 'requires' is deprecated, install_requires is preferred
)
