from setuptools import setup, find_packages

setup(
    name='PythonAI',
    version='0.1.0',
    description='Make AI Brain with pyai',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Divyanshu Sinha',
    author_email='divyanshu.sinha136@gmail.com',
    url='https://www.github.com/World-Of-Programming-And-Technology/PythonAI',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Lesser General Public License v2.1',
    ],
    python_requires='>=3.6',
)
