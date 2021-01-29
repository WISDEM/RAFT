import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FrequencyDomain",
    version="0.1",
    author="author",
    author_email="author@address.com",
    description="RAFT: Response Amplitudes of Floating Turbines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattEhall/FrequencyDomain",
    packages=['raft', 'hams'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
