from setuptools import setup, find_packages

setup(
    name="ebmcmc",
    version="0.1.0",  # Update version as needed
    author="Jackie Blaum",
    author_email="your.email@example.com",  # Replace with your email
    description="A package for MCMC analysis of eclipsing binaries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jackieblaum/ebmcmc",
    packages=find_packages(),  
    install_requires=[
        "numpy>=1.21",     
        "pandas>=1.3",       
        "matplotlib>=3.4",   
        "astropy>=5.0",    
        "scipy>=1.7",       
        "emcee>=3.1",       
        "tqdm>=4.62",        
        "pymc>=5.0",        
        "aesara>=2.7"       
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Use your license type
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  # Specify Python version compatibility
)
