from setuptools import setup, find_packages

setup(
    name='volatility-forecast',
    version='0.1.0',
    author='Steve Yang',
    author_email='steveya@gmail.com',
    description='A package of volatility forecasting models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/steveta/volatility-forecast',
    packages=find_packages(where='volatility_forecast'),
    package_dir={'': 'volatilty_forecast'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy'
    ],
)