'''
Copyright © 2024 Meta Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code to run the compiled soft shrink kernel.
'''

# Allow soft shrink op to run through CPU fallback if it's not implemented.
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch.utils.cpp_extension

fp32_compiled_lib = torch.utils.cpp_extension.load(
    name='LlamaCppFloatLinear',
    sources=['LlamaCppFloatLinear.mm'],
    extra_include_paths=[os.path.dirname(__file__)],
    extra_cflags=['-std=c++17', "-g", "-D_CAPTURE_KERNEL=1", "-DDEBUG=1"],
    verbose=True,
   )

int8_compiled_lib = torch.utils.cpp_extension.load(
    name='LlamaCppInt8Linear',
    sources=['LlamaCppInt8Linear.mm'],
    extra_include_paths=[os.path.dirname(__file__)],
    extra_cflags=['-std=c++17', "-g", "-D_CAPTURE_KERNEL=1", "-DDEBUG=1"],
    verbose=True,
   )

if __name__ == "__main__":
    mps_device = torch.device("mps")  # Device object representing GPU.
    weight = torch.randn(1024, 4096, device=mps_device, dtype=torch.float32)
    input = torch.randn(2, 4096, device=mps_device, dtype=torch.float32)
    print("Running compiled kernel")
    res1 = fp32_compiled_lib.llama_cpp_mm_f32(input, weight)
    res2 = torch.mm(input, weight.transpose(1, 0).contiguous())

    print("FP32 MPS result: ")
    print(res2)

    print("FP32 Metal kernel result: ")
    print(res1)

    print(f"Allclose? {torch.allclose(res1, res2, atol=0.1)}")

    qweight = torch.randn(1024, 4096, device=mps_device, dtype=torch.int8)
    scale = torch.randn(1024, device=mps_device, dtype=torch.float32)
    res1 = int8_compiled_lib.llama_cpp_mm_i8(input, qweight, scale)
    res2 = torch._weight_int8pack_mm(input, qweight, scale)

    print("Scale: ")
    print(scale)
    print("Int8 MPS result: ")
    print(res2)

    print("Int8 Metal kernel result: ")
    print(res1)

    print(f"Allclose? {torch.allclose(res1, res2, atol=0.1)}")