import torch

def check_gpu_support():
    print("=== GPU加速支持检测 ===")
    
    # 检测CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA支持: {'是' if cuda_available else '否'}")
    
    if cuda_available:
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"检测到GPU数量: {gpu_count}")
        
        # 获取每个GPU的详细信息
        for i in range(gpu_count):
            print(f"\nGPU {i} 详细信息:")
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"名称: {gpu_props.name}")
            print(f"计算能力: {gpu_props.major}.{gpu_props.minor}")
            print(f"显存总量: {gpu_props.total_memory / 1024**3:.2f} GB")
            print(f"多处理器数量: {gpu_props.multi_processor_count}")
    else:
        print("\n未检测到支持CUDA的GPU设备")
    
    # 检查是否支持MPS (Apple Silicon GPU)
    try:
        mps_available = torch.backends.mps.is_available()
        print(f"\nApple MPS (Metal)支持: {'是' if mps_available else '否'}")
        if mps_available:
            print("检测到Apple Silicon GPU")
    except AttributeError:
        print("\nApple MPS (Metal)支持: 不可用(非Mac系统或PyTorch版本过低)")

if __name__ == "__main__":
    check_gpu_support()
