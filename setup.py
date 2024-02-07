
import hmesh
import setuptools

with open('README.md', 'r') as f:
    long_descriprion = f.read()

setuptools.setup(
    name='hmesh',
    version=hmesh.__version__,
    author='helmholtz',
    author_email='helmholtz@fomal.host',
    description='hmesh - CUDA mesh utilities for trimesh and torch',
    long_description=long_descriprion,
    long_descriprion_content_type='text/markdown',
    url='',
    packages=setuptools.find_packages(),
    package_data={
        'hmesh': [
            'backend/*.cpp',
            'backend/*.h',
            'backend/*.py',
            'backend/*.cu',
            'ray/*.py',
            '*.py'
        ]
    },
    include_package_data=True,
    install_requires=['trimesh', 'torch', 'jaxtyping'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.9'
)
