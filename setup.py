from setuptools import setup, find_packages

setup(
    name="raytracing-one-weekend",
    version="0.0.0",
    author="Andy Palmer",
    author_email="contactninezerozeronine@gmail.com",
    description="A raytracer achievable in a weekend.",
    url="https://github.com/ninezerozeronine/raytracing-one-weekend",
    install_requires=[
        "Pillow",
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
