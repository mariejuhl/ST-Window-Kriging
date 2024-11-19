from setuptools import setup, find_packages

setup(
    name="my-package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',  # Example of dependency
        'pandas',  # Example of dependency
    ],
    author="Your Name",
    author_email="your_email@example.com",
    description="A brief description of your package",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/my-package",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
