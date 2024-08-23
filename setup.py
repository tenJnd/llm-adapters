import os

from setuptools import setup, find_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


version = os.environ.get('CI_COMMIT_TAG', f"1.0")

setup(
    name='llm-adapters',
    version=version,
    description='Simple llm adapters for opnAi and local models',
    long_description=readme(),
    url='https://github.com/tenJnd/llm-adapters',
    author='Tomas.jnd',
    author_email='',
    packages=find_packages(exclude=('tests', 'docs')),
    python_requires='>=3.6',
    install_requires=[
        'backoff==2.2.1',
        'huggingface-hub==0.24.6',
        'llama_cpp_python==0.2.77',
        'lxml==5.3.0',
        'numpy==2.1.0',
        'openai==1.42.0',
        'requests==2.32.3',
        'tiktoken==0.7.0',
        'tokenizers==0.19.1',
        'tqdm==4.66.5',
        'transformers==4.44.2'
    ],
)
