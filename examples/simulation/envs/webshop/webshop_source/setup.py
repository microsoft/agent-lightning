from setuptools import setup, find_packages

setup(
    name='webshop',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        "beautifulsoup4==4.11.1",
        "cleantext==1.1.4",
        "env==0.1.0",
        "faiss-cpu==1.7.4",
        "Flask==2.1.2",
        "gym==0.24.0",
        "pyserini==0.17.0",
        "pytest",
        "rank_bm25==0.2.2",
        "requests_mock",
        "scikit_learn==1.1.1",
        "selenium==4.2.0",
        "spacy==3.6.1",
        "thinc==8.1.12",
        "thefuzz==0.20.0",
        "werkzeug==2.3.8",
    ],
    author='Your Name',
    author_email='youremail@example.com',
    description='webshop pip package version',
    license='MIT',
    keywords='sample setuptools development',
    url='https://github.com/yourusername/mypackage'
)