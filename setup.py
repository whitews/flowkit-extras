from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

reqs = [
    'flowio',
    'flowkit',
    'numpy',
    'pandas',
    'sklearn',
    'scipy',
    'matplotlib',
    'seaborn',
    'statsmodels'
]

# on_rtd = os.environ.get('READTHEDOCS') == 'True'
# if on_rtd:
#     reqs.remove('multicoretnse')

setup(
    name='flowkit_extras',
    version='0.1b',
    packages=find_packages(),
    description='Extra functions for the Flow Cytometry Toolkit (FlowKit)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Scott White",
    license='BSD',
    url="https://github.com/whitews/flowkit-extras",
    ext_modules=[],
    install_requires=reqs,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6'
    ]
)
