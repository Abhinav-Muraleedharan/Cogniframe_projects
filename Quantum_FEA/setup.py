from setuptools import setup, find_packages

setup(
    name='Quantum-FEA',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # 'numpy',
        # 'scipy',
    ],
    author='Abhinav Muraleedharan',
    author_email='abhinav@cogniframe.com',
    description='Code for Quantum FEA project',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved ::Commercial License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
