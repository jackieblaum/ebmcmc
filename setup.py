from setuptools import setup, find_packages

setup(
    name="ebmcmc",
    version="0.1.0",
    author="Jackie Blaum",
    author_email="jrblaum@berkeley.edu",
    description="MCMC fitting of eclipsing binary star systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jackieblaum/ebmcmc",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "astropy>=5.0",
        "emcee>=3.1",
        "h5py>=3.0",
        "joblib>=1.0",
        "tqdm>=4.62",
        "dustmaps>=1.0",
        "matplotlib>=3.4",
    ],
    extras_require={
        "backends": [
            "phoebe>=2.4",
            "binarysed",
            "ellc",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.9",
)
