from setuptools import setup, find_packages

setup(
    name='thermal_sim',
    version='0.1.0',
    description='Extensible thermal system simulator for coupled thermo-hydraulic analysis',
    author='Thermal Simulation Team',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'CoolProp>=6.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ]
    },
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
