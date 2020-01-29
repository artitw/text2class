import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text2class",
    version="0.0.4",
    author="Artit Wangperawong",
    author_email="artitw@gmail.com",
    description="Multi-class text categorization using state-of-the-art pre-trained contextualized language models, e.g. BERT.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artitw/text2class",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords='bert nlp text classification data science machine learning',
    install_requires=[
        'tensorflow==1.15.2',
        'tensorflow_hub',
        'pandas',
    ],
)
