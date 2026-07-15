from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anglepy",
    version="0.0.1",
    author="Rajdeep Pathak, Archi Roy, Tanujit Chakraborty",
    author_email="pathakrajdeep91@gmail.com",
    description="ANGLE: Angular Neural Generative Learning via Engression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Circular Data Analysis",
    ],
    project_urls={
        "Documentation": "https://anglepy.readthedocs.io/",
        "Source Code": "https://github.com/PyCoder913/anglepy",
        "Paper": "https://arxiv.org/abs/2607.12833",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.7.0",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "astropy"
    ],
)
