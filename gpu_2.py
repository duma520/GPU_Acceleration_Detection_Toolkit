import torch
import platform

def check_gpu_acceleration():
    print("=== 本机 GPU 加速支持检测 ===")
    system = platform.system()
    
    # 检测 CUDA (NVIDIA GPU)
    cuda_available = torch.cuda.is_available()
    print(f"\n[1] CUDA (NVIDIA GPU) 支持: {'✅ 可用' if cuda_available else '❌ 不可用'}")
    if cuda_available:
        print(f"    - GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"    - GPU {i}: {gpu_name}")

    # 检测 ROCm (AMD GPU)
    try:
        rocm_available = torch.version.hip is not None
        print(f"\n[2] ROCm (AMD GPU) 支持: {'✅ 可用' if rocm_available else '❌ 不可用'}")
    except:
        print("\n[2] ROCm (AMD GPU) 支持: ❌ 不可用 (PyTorch 未编译 ROCm 支持)")

    # 检测 Metal (Apple Silicon)
    try:
        mps_available = torch.backends.mps.is_available()
        print(f"\n[3] Metal (Apple M1/M2/M3) 支持: {'✅ 可用' if mps_available else '❌ 不可用'}")
    except:
        print("\n[3] Metal (Apple M1/M2/M3) 支持: ❌ 不可用 (非 macOS 或 PyTorch 版本过低)")

    # 检测 OpenCL (跨平台)
    try:
        import pyopencl
        print("\n[4] OpenCL 支持: ✅ 可用 (需安装 `pyopencl`)")
    except ImportError:
        print("\n[4] OpenCL 支持: ❌ 不可用 (未安装 `pyopencl`)")

    # 检测 Intel oneAPI (Intel GPU)
    try:
        import dpctl
        print("\n[5] Intel oneAPI (Intel GPU) 支持: ✅ 可用 (需安装 `dpctl`)")
    except ImportError:
        print("\n[5] Intel oneAPI (Intel GPU) 支持: ❌ 不可用 (未安装 `dpctl`)")

if __name__ == "__main__":
    check_gpu_acceleration()
