from setuptools import setup, find_packages


setup(
    name='nanogpt',
    version='0.0.1',
    description='Use deep learning to train language models',
    url='https://github.com/daniel-layton-wright/nanogpt.git',
    author='Daniel Wright',
    author_email='dlwright@alumni.stanford.edu',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    package_data={
        'nanogpt': [
        ]
    },
    install_requires=[
        'torch', 'numpy', 'lightning', 'hydra-core', 'wandb'
    ],
    tests_require=[],
    setup_requires=[],
    zip_safe=True,
)
