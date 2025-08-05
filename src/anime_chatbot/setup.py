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
