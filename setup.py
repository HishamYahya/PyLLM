from setuptools import setup, find_packages

version = "0.0.2"

with open("README.md") as f:
    long_description = f.read()

setup(
    name="pyllm-code",
    version=version,
    author="Hisham Alyahya",
    author_email="Hishamaalyahya@gmail.com",
    license="MIT",
    description="Leverage LLMs to generate and execute robust code dynamically through an intuitive and easy-to-use API!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "LLM",
        "LLMs",
        "assistant",
        "helper",
        "coding",
        "automation",
        "AI",
        "NLP",
    ],
    download_url=f"https://github.com/HishamYahya/PyLLM/archive/refs/tags/v{version}.tar.gz",
    packages=find_packages(),
    url="https://github.com/HishamYahya/PyLLM",
    install_requires=["filelock", "Jinja2", "Requests", "appdirs", "pytest"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
