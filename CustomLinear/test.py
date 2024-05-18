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

compiled_lib = torch.utils.cpp_extension.load(
    name='Int8PackedLinear',
    sources=['LlamaCppInt8Linear.mm'],
    extra_cflags=['-std=c++17', "-g", "-D_CAPTURE_KERNEL=1", "-DDEBUG=1"],
    verbose=True,
   )


if __name__ == "__main__":
    mps_device = torch.device("mps")  # Device object representing GPU.
    # Create a tensor with values from 0 to 127
    row = torch.arange(32, device=mps_device, dtype=torch.int8)
    # Repeat the row 64 times to create a 64x128 tensor
    weight = row.repeat(64, 1)
    input_row = torch.arange(32, device=mps_device, dtype=torch.float32)
    input = input_row.repeat(32, 1)
    scale = torch.randn(64, device=mps_device, dtype=torch.float32)
    print(scale)
    print("Running compiled kernel")
    res1 = compiled_lib.llama_cpp_mps_int8_linear(input, weight, scale)
    res2 = torch.ops.aten._weight_int8pack_mm(input, weight, scale)
    print("Metal kernel result: ")
    print(res1)

    print("MPS result: ")
    print(res2)

    print(f"Allclose? {torch.allclose(res1, res2, atol=1e-4)}")
