from setuptools import setup

setup(
    name='pytorch-vae-models',
    version='0.1.0',
    packages=['vae_models', 'configs', 'diffusion', 'tests'],
    install_requires=[
        'torch',
        'torchvision',
        'pytorch-lightning',
        'numpy',
        'PyYAML',
        'Pillow',
    ],
)