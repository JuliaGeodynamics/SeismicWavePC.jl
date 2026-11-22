# Pick the KernelAbstractions backend from the first command-line argument: cpu (default), cuda, amdgpu or metal.
# The GPU package must be installed in the active environment (e.g. `] add CUDA`).
const DEVICE = isempty(ARGS) ? "cpu" : lowercase(ARGS[1])
if DEVICE == "cuda"
    using CUDA
elseif DEVICE == "amdgpu"
    using AMDGPU
elseif DEVICE == "metal"
    using Metal
end
backend = DEVICE == "cuda" ? CUDABackend() : DEVICE == "amdgpu" ? ROCBackend() : DEVICE == "metal" ? MetalBackend() : CPU()
