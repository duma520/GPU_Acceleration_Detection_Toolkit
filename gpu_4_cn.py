# GPU与加速支持检测工具 v1.2
import sys
import platform
import subprocess
import pkg_resources
import importlib
import torch
import tensorflow as tf
from numba import cuda

__version__ = "1.2"

# 修改cupy导入方式，添加异常处理
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("警告: CuPy未安装或安装不正确，部分功能将不可用")
    print("建议安装命令: pip install cupy-cuda11x (请将11x替换为您的CUDA版本)")

# 添加tabulate的异常处理
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("警告: tabulate未安装，表格输出将使用简单格式")
    print("建议安装命令: pip install tabulate")

import cv2
import numpy as np
from PIL import Image
import psutil
import multiprocessing

def detect_gpu_support():
    """检测系统中可用的GPU和加速支持"""
    results = []
    
    # 1. 系统基本信息
    system_info = {
        "操作系统": platform.system(),
        "硬件架构": platform.machine(),
        "处理器": platform.processor(),
        "Python版本": sys.version.split()[0],
        "CPU核心数": multiprocessing.cpu_count(),
        "总内存": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "检测工具版本": __version__
    }
    results.append(("系统基本信息", system_info))
    
    # 2. 检测CUDA支持
    cuda_support = {}
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        cuda_support["CUDA编译器"] = nvcc_version.split('\n')[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        cuda_support["CUDA编译器"] = "未找到"
    
    try:
        import pycuda.driver as drv
        drv.init()
        cuda_support["PyCUDA设备"] = [f"{drv.Device(i).name()}" for i in range(drv.Device.count())]
    except ImportError:
        cuda_support["PyCUDA设备"] = "PyCUDA未安装"
    except Exception as e:
        cuda_support["PyCUDA设备"] = f"错误: {str(e)}"
    
    results.append(("CUDA支持情况", cuda_support))
    
    # 3. 检测OpenCV GPU支持
    opencv_gpu = {}
    try:
        opencv_gpu["OpenCV版本"] = cv2.__version__
        opencv_gpu["CUDA支持"] = "是" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "否"
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            opencv_gpu["CUDA设备"] = [f"设备{i}: {cv2.cuda.Device(i).name()}" for i in range(cv2.cuda.getCudaEnabledDeviceCount())]
    except Exception as e:
        opencv_gpu["OpenCV GPU支持"] = f"检测错误: {str(e)}"
    
    results.append(("OpenCV GPU支持", opencv_gpu))
    
    # 4. 检测PyTorch GPU支持
    pytorch_gpu = {}
    try:
        pytorch_gpu["PyTorch版本"] = torch.__version__
        pytorch_gpu["CUDA可用"] = "是" if torch.cuda.is_available() else "否"
        if torch.cuda.is_available():
            pytorch_gpu["CUDA设备"] = [f"设备{i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
            pytorch_gpu["当前设备"] = torch.cuda.current_device()
            pytorch_gpu["CUDA版本"] = torch.version.cuda
    except Exception as e:
        pytorch_gpu["PyTorch GPU支持"] = f"检测错误: {str(e)}"
    
    results.append(("PyTorch GPU支持", pytorch_gpu))
    
    # 5. 检测TensorFlow GPU支持
    tf_gpu = {}
    try:
        tf_gpu["TensorFlow版本"] = tf.__version__
        gpu_devices = tf.config.list_physical_devices('GPU')
        tf_gpu["GPU可用"] = "是" if gpu_devices else "否"
        if gpu_devices:
            tf_gpu["GPU详情"] = [f"{gpu.name} - {gpu.device_type}" for gpu in gpu_devices]
    except Exception as e:
        tf_gpu["TensorFlow GPU支持"] = f"检测错误: {str(e)}"
    
    results.append(("TensorFlow GPU支持", tf_gpu))
    
    # 6. 检测Numba GPU支持
    numba_gpu = {}
    try:
        numba_gpu["Numba版本"] = importlib.metadata.version('numba')
        numba_gpu["CUDA可用"] = "是" if cuda.is_available() else "否"
        if cuda.is_available():
            numba_gpu["CUDA设备"] = [f"设备{i}: {cuda.gpus[i].name}" for i in range(len(cuda.gpus))]
    except Exception as e:
        numba_gpu["Numba GPU支持"] = f"检测错误: {str(e)}"
    
    results.append(("Numba GPU支持", numba_gpu))
    
    # 7. 检测CuPy支持
    cupy_gpu = {}
    if CUPY_AVAILABLE:
        try:
            cupy_gpu["CuPy版本"] = cp.__version__
            cupy_gpu["CUDA支持"] = "是"  # CuPy必须依赖CUDA
            cupy_gpu["CUDA设备"] = [f"设备{i}: {cp.cuda.runtime.getDeviceProperties(i)['name']}" 
                                  for i in range(cp.cuda.runtime.getDeviceCount())]
        except Exception as e:
            cupy_gpu["CuPy支持"] = f"检测错误: {str(e)}"
    else:
        cupy_gpu["CuPy支持"] = "未安装或安装不正确"
        cupy_gpu["安装建议"] = "请使用 pip install cupy-cuda11x 安装(替换11x为您的CUDA版本)"
    
    results.append(("CuPy支持情况", cupy_gpu))
    
    # 8. 检测其他加速库
    other_accel = {}
    
    # Dask并行计算
    try:
        import dask
        other_accel["Dask版本"] = dask.__version__
        try:
            from dask.distributed import Client
            client = Client()
            other_accel["Dask工作节点"] = len(client.ncores())
            client.close()
        except:
            other_accel["Dask工作节点"] = "未初始化"
    except ImportError:
        other_accel["Dask"] = "未安装"
    
    # Numba JIT即时编译
    try:
        from numba import jit
        other_accel["Numba JIT"] = "可用"
    except ImportError:
        other_accel["Numba JIT"] = "未安装"
    
    # Cython加速
    try:
        import Cython
        other_accel["Cython版本"] = Cython.__version__
    except ImportError:
        other_accel["Cython"] = "未安装"
    
    # MPI并行
    try:
        import mpi4py
        other_accel["MPI支持"] = "通过mpi4py可用"
    except ImportError:
        other_accel["MPI支持"] = "未安装"
    
    # Ray分布式计算
    try:
        import ray
        other_accel["Ray版本"] = ray.__version__
    except ImportError:
        other_accel["Ray"] = "未安装"
    
    results.append(("其他加速库支持", other_accel))
    
    return results

def print_detection_results(results):
    """打印中文检测结果"""
    print("\n" + "="*80)
    print("GPU与Python加速支持检测报告 (v{})".format(__version__))
    print("="*80 + "\n")
    
    for category, data in results:
        print(f"\n{category}:")
        print("-"*len(category))
        
        if isinstance(data, dict):
            if TABULATE_AVAILABLE:
                table_data = [(k, v) for k, v in data.items()]
                print(tabulate(table_data, headers=["检测项", "状态"], tablefmt="grid"))
            else:
                for k, v in data.items():
                    print(f"{k}: {v}")
        else:
            print(data)
        
        print()

def check_installation_commands():
    """生成中文推荐安装命令"""
    commands = {
        "PyTorch": "pip install torch torchvision torchaudio",
        "TensorFlow": "pip install tensorflow",
        "TensorFlow GPU版": "pip install tensorflow-gpu",
        "OpenCV (带CUDA支持)": "pip install opencv-python-headless opencv-contrib-python-headless",
        "Numba": "pip install numba",
        "CuPy": "pip install cupy-cuda11x (将11x替换为你的CUDA版本)",
        "Dask": "pip install dask distributed",
        "Cython": "pip install cython",
        "MPI支持": "pip install mpi4py",
        "Ray": "pip install ray",
        "Tabulate": "pip install tabulate"  # 新增tabulate安装建议
    }
    
    print("\n推荐安装命令:")
    print("-"*30)
    for lib, cmd in commands.items():
        print(f"{lib}: {cmd}")

def main():
    print("正在检测GPU和加速支持...")
    results = detect_gpu_support()
    print_detection_results(results)
    check_installation_commands()
    
    # 最终建议
    print("\n" + "="*80)
    print("优化建议:")
    print("- 如果显示GPU支持为'否'，请检查显卡驱动和CUDA安装")
    print("- 对于深度学习开发，推荐安装PyTorch或TensorFlow的GPU版本")
    print("- 对于数值计算加速，可以考虑使用Numba或CuPy")
    print("- 对于并行计算，Dask和Ray是不错的选择")
    print("- 如需更好的表格显示效果，请安装tabulate模块")
    print("="*80)

if __name__ == "__main__":
    main()
