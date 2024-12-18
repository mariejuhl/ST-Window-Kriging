from setuptools import setup, find_packages

setup(
    name="kriging_tools",
    #version="",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here, e.g., 'numpy'
    ],
    author="M.-C.Juhl",
    author_email="mariechristin.juhl@tum.de",
    description="Tools needed in the CIRCOS-project: Estimating Correlation and coherence of gridded data; perform different forms of Kriging and Co-Kriging; Validation of Grids against tide gauges ",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url = "https://gitlab.lrz.de/00000000014B1C62/kriging-tools/", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
