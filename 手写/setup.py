from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="multi_head_attn",
    ext_modules=[
        CUDAExtension(
            "multi_head_attn",
            sources=["multi_head_attn.cu"],
            extra_compile_args={
                "nvcc": ["-O3", "-arch=sm_90", "-allow-unsupported-compiler" ]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)