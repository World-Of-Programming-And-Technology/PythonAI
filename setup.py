from setuptools import setup, find_packages

setup(
    name='PythonAIBrain',  
    version='0.1.0',  
    packages=find_packages(), 
    install_requires=[
        'nltk',
        'pyjokes',
        'yfinance',
        'numpy',
        'scikit-learn',
        'torch',
    ],
    author='Divyanshu Sinha',
    description='Create AI with PythonAIBrain on using its Brain class.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.7', 
)
