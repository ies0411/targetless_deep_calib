import ast
import os
import re

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def version(name):
    _re = re.compile(r"__version__\s+=\s+(.*)")
    with open(f"__init__.py", "rb") as f:
        return str(ast.literal_eval(_re.search(f.read().decode("utf-8")).group(1)))


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name="%s.%s" % (module, name),
        sources=[os.path.join(*module.split("."), src) for src in sources],
    )
    return cuda_ext


setup(
    name="targetless",
    version=version("targetless"),
    author="Eunsoo Im",
    author_email="eslim@superb-ai.com",
    packages=find_packages(exclude=["tools", "scripts"]),
    install_requires=[
        "easydict==1.10",
        "pyyaml==5.4.1",
        "SharedArray==3.2.2",
        "pyquaternion==0.9.9",
        "filterpy==1.4.5",
        "open3d==0.17.0",
        "kornia==0.6.11",
        "filelock>=3.0.12",
        "natsort>=7.1.1",
        "numpy>=1.18.5",
        "opencv-python==4.4.0.46",
        "opencv-contrib-python==4.4.0.46",
        "pynvml>=8.0.4",
        "scipy>=1.5.4",
        "toml>=0.10.2",
        "timm>=0.4.12",
        "mypy_extensions",
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    ext_modules=[
        make_cuda_ext(
            name="correlation_cuda",
            module="models.custom",
            sources=[
                "src/correlation_cuda_kernel.cu",
                "src/correlation_cuda.cc",
            ],
        ),
    ],
)
