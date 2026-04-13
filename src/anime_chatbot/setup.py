"""
Package setup configuration for the Anime Chatbot.

This module defines the package metadata, dependencies, and installation
configuration for the anime_chatbot package using setuptools. It specifies
required packages, test dependencies, and package discovery settings.
"""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'requests>=2.32.5',
]

TEST_EXTRAS = [
    'pytest>=8.0.0',
]

setup(
    name='anime_chatbot',
    version='0.2',
    author='WaifuAI',
    author_email='waifuai@users.noreply.github.com',
    url='https://github.com/waifuai/anime-subtitle-chatbot-trax',
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        'test': TEST_EXTRAS,
    },
    packages=find_packages(),
    include_package_data=True,
    description='Anime Subtitle Chatbot with OpenRouter AI Provider',
    requires=[]
)
