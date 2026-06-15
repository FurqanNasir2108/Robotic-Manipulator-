from setuptools import setup, find_packages

setup(
    name='three_link_manipulator',
    version='0.1.0',
    description='Generative Models for Trajectory Generation in a 3-Link Planar Robotic Manipulator',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.12,<2.13',
        'numpy>=2.4,<2.5',
        'scipy>=1.17,<1.18',
        'matplotlib>=3.10,<3.11',
        'seaborn>=0.13,<0.14',
        'PyYAML>=6.0,<7.0',
        'h5py>=3.16,<3.17',
        'tensorboard>=2.20,<2.21',
        'tqdm>=4.68,<4.69',
        'scikit-learn>=1.9,<1.10',
        'onnx>=1.21,<1.22',
        'onnxscript>=0.7,<0.8',
        'psutil>=7.2,<8.0',
    ],
    extras_require={
        'dev': [
            'pytest>=9.0,<10.0',
        ],
        'edge': [
            'onnxruntime>=1.26,<2.0',
            'Flask>=3.1,<4.0',
            'paho-mqtt>=2.1,<3.0',
        ],
    },
    python_requires='>=3.10',
)
