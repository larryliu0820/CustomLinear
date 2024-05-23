Run `cd CustomLinear; MTL_CAPTURE_ENABLE=1 python test.py`

Here's an example output:

```
❯ MTL_CAPTURE_ENABLED=1 python test.py
Using /Users/larryliu/Library/Caches/torch_extensions/py311_cpu as PyTorch extensions root...
Emitting ninja build file /Users/larryliu/Library/Caches/torch_extensions/py311_cpu/LlamaCppFloatLinear/build.ninja...
Building extension module LlamaCppFloatLinear...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/2] c++ -MMD -MF LlamaCppFloatLinear.o.d -DTORCH_EXTENSION_NAME=LlamaCppFloatLinear -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_clang\" -DPYBIND11_STDLIB=\"_libcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1002\" -I/Users/larryliu/Desktop/CustomLinear/CustomLinear -isystem /Users/larryliu/miniconda3/envs/executorch/lib/python3.11/site-packages/torch/include -isystem /Users/larryliu/miniconda3/envs/executorch/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /Users/larryliu/miniconda3/envs/executorch/lib/python3.11/site-packages/torch/include/TH -isystem /Users/larryliu/miniconda3/envs/executorch/lib/python3.11/site-packages/torch/include/THC -isystem /Users/larryliu/miniconda3/envs/executorch/include/python3.11 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -std=c++17 -g -D_CAPTURE_KERNEL=1 -DDEBUG=1 -c /Users/larryliu/Desktop/CustomLinear/CustomLinear/LlamaCppFloatLinear.mm -o LlamaCppFloatLinear.o 
In file included from /Users/larryliu/Desktop/CustomLinear/CustomLinear/LlamaCppFloatLinear.mm:20:
/Users/larryliu/Desktop/CustomLinear/CustomLinear/CustomLinear.h:12:33: warning: ISO C++11 does not allow conversion from string literal to 'char *' [-Wwritable-strings]
static char* QUANTIZED_KERNEL = R"METAL_QUANTIZED(
                                ^
1 warning generated.
[2/2] c++ LlamaCppFloatLinear.o -shared -L/Users/larryliu/miniconda3/envs/executorch/lib/python3.11/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -undefined dynamic_lookup -o LlamaCppFloatLinear.so
Loading extension module LlamaCppFloatLinear...
2024-05-22 12:22:44.973 python[52068:873525] Metal GPU Frame Capture Enabled
Running compiled kernel
MPS result: 
tensor([[  -3.2085,    9.3427,  -93.3763,  ...,  -42.1728,   81.4455,
           57.8988],
        [-133.8210,   79.5968,   32.0507,  ...,  -83.1579,   71.9521,
          -17.6302]], device='mps:0')
Metal kernel result: 
tensor([[  -3.2113,    9.3391,  -93.3791,  ...,  -42.1828,   81.4648,
           57.9186],
        [-133.8141,   79.6262,   32.0712,  ...,  -83.1528,   71.9571,
          -17.6362]], device='mps:0')
Allclose? True
```
