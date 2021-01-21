import setuptools

setuptools.setup(
    name="perception", # Replace with your own username
    version="0.0.1",
    author="Kenneth Blomqvist",
    author_email="hello@keke.dev",
    description="A collection of utilities for doing robotic perception in Python and ROS.",
    url="https://github.com/kekeblom/perception",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License"
    ],
    python_requires='>=3.6',
)

