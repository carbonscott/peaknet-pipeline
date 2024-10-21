import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="peaknet-pipeline",
    version="24.10.14",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="Save PeakNet inference results to CXI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/peaknet-pipeline",
    keywords = ['SFX',],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts' : [
            'peaknet-pipeline-mpi=peaknet_pipeline.run_mpi:main',
            'peaknet-pipeline-ray=peaknet_pipeline.run_ray:main',
            'peaknet-pipeline-write-to-cxi=peaknet_pipeline.cxi_consumer:main',
        ],
    },
    python_requires='>=3.6',
    include_package_data=True,
)
