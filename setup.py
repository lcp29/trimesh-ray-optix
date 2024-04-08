
import os
import triro
import subprocess
import setuptools

if os.name == 'nt':
    from setuptools import msvc

with open('README.md', 'r', encoding='utf-8') as f:
    long_descriprion = f.read()

SHADER_LOCATION = 'triro/backend/shaders.cu'
SHADER_DIR = os.path.dirname(SHADER_LOCATION)
SHADER_BASENAME = os.path.basename(SHADER_LOCATION)
SHADER_WORK_DIR = os.path.join(SHADER_DIR, 'embedded')
SHADER_IR_PATH = os.path.join(SHADER_WORK_DIR, SHADER_BASENAME[:SHADER_BASENAME.rfind('.')] + '.optixir')
SHADER_HEADER_PATH = os.path.join(SHADER_WORK_DIR, SHADER_BASENAME[:SHADER_BASENAME.rfind('.')] + '_embedded.h')
SHADER_TEMPLATE_PATH = os.path.join(SHADER_WORK_DIR, 'shader_template.h')

def compile_and_embed_shaders():
    optix_include_dir = os.path.join(os.environ['OptiX_INSTALL_DIR'], 'include')
    # compile shader files
    if not os.path.exists(SHADER_WORK_DIR):
        os.makedirs(SHADER_WORK_DIR)
    # compileer options, see OptiX Programming Guide 6.1
    compile_command = [
        'nvcc',
        '-optix-ir',
        f'-I{optix_include_dir}',
        '--use_fast_math',
        '-m64',
        '--relocatable-device-code=true',
        '--std=c++17',
        '--expt-relaxed-constexpr', # for <tuple> to be correctly compiled
        SHADER_LOCATION,
        '-o',
        SHADER_IR_PATH
    ]
    # should specify correct C++ compiler on Windows
    if os.name == 'nt':
        msvc_env = msvc.EnvironmentInfo('amd64')
        msvc_dir_candidates = msvc_env.VCTools
        for d in msvc_dir_candidates:
            if 'bin' in d:
                msvc_dir = d
                break
        compile_command += ['-ccbin', msvc_dir]

    subprocess.call(compile_command)

    # embed shaders
    with open(SHADER_IR_PATH, 'rb') as f:
        shader_hex = f.read().hex()
    
    shader_formatted = ''
    for i in range(0, len(shader_hex), 2):
        shader_formatted += f'0x{shader_hex[i:i+2]}, ' + ('\n' if i % 16 == 14 else '')
    
    shader_formatted = shader_formatted.strip(' ,\n')
    shader_length = len(shader_hex) // 2
    
    with open(SHADER_TEMPLATE_PATH, 'r') as f:
        shader_template = f.read()

    shader_template = shader_template \
        .replace('SHADER_LENGTH', str(shader_length)) \
        .replace('SHADER_FORMATTED', shader_formatted)
    
    with open(SHADER_HEADER_PATH, 'w+') as f:
        f.write(shader_template)

compile_and_embed_shaders()

setuptools.setup(
    name='triro',
    version=triro.__version__,
    author='helmholtz',
    author_email='helmholtz@fomal.host',
    description='Triro - An in-place replacement for trimesh.ray in Optix',
    long_description=long_descriprion,
    long_descriprion_content_type='text/markdown',
    url='',
    packages=setuptools.find_packages(),
    package_data={
        'triro': [
            'backend/embedded/shaders_embedded.h',
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
