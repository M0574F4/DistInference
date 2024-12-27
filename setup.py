# setup.py

from setuptools import setup, find_packages

setup(
    name='DistInference',
    version='0.1.0',
    author='Mostafa Naseri',
    author_email='mostafa.naseri@ugent.be',
    description='Activity Recognition via Distributed Inference Project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/M0574F4/DistInference',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
