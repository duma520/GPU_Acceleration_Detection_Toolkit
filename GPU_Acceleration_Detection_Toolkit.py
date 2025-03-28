import torch
import platform
import sys
import subprocess
import pkg_resources
import importlib
import tensorflow as tf
from numba import cuda
import cv2
import numpy as np
from PIL import Image
import psutil
import multiprocessing
import datetime
import json
from typing import Dict, List, Optional

__version__ = "4.1"
author = "杜玛"
copyrigh = "Copyright © 杜玛. All rights reserved."

# 检查可选依赖项
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

class GPUAccelerationDetector:
    def __init__(self):
        self.system_info = self._collect_system_info()
        self.acceleration_methods = {}
        self.report_filename = f"gpu_acceleration_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def _collect_system_info(self) -> Dict:
        """收集系统基本信息"""
        return {
            "操作系统": platform.system(),
            "硬件架构": platform.machine(),
            "处理器": platform.processor(),
            "Python版本": sys.version.split()[0],
            "PyTorch版本": torch.__version__,
            "TensorFlow版本": tf.__version__ if 'tf' in globals() else "未安装",
            "OpenCV版本": cv2.__version__,
            "CPU核心数": multiprocessing.cpu_count(),
            "总内存": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "检测工具版本": __version__
        }
    
    def detect_all(self) -> Dict:
        """检测所有可用的加速方案"""
        self._detect_cuda()
        self._detect_rocm()
        self._detect_metal()
        self._detect_opencl()
        self._detect_intel_oneapi()
        self._detect_opencv_acceleration()
        self._detect_numba()
        self._detect_tensorflow_gpu()
        self._detect_cupy()
        self._detect_other_acceleration_libs()
        
        return {
            "系统信息": self.system_info,
            "加速方案": self.acceleration_methods,
            "优化建议": self._generate_optimization_advice()
        }
    
    def _detect_cuda(self):
        """检测NVIDIA CUDA支持"""
        cuda_info = {"可用": False}
        try:
            if torch.cuda.is_available():
                cuda_info["可用"] = True
                cuda_info["设备数量"] = torch.cuda.device_count()
                cuda_info["设备列表"] = []
                
                for i in range(torch.cuda.device_count()):
                    device_info = {
                        "名称": torch.cuda.get_device_name(i),
                        "计算能力": f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}",
                        "显存": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                    }
                    cuda_info["设备列表"].append(device_info)
                
                # 检测OpenCV CUDA支持
                try:
                    cuda_info["opencv_cuda"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
                except:
                    cuda_info["opencv_cuda"] = False
                
                # 检测NVIDIA编译器
                try:
                    nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
                    cuda_info["CUDA编译器"] = nvcc_version.split('\n')[0]
                except:
                    cuda_info["CUDA编译器"] = "未找到"
        except:
            pass
        
        self.acceleration_methods["CUDA"] = cuda_info
    
    def _detect_rocm(self):
        """检测AMD ROCm支持"""
        rocm_info = {"可用": False}
        try:
            rocm_info["可用"] = torch.version.hip is not None
            if rocm_info["可用"]:
                rocm_info["版本"] = torch.version.hip
        except:
            pass
            
        self.acceleration_methods["ROCm"] = rocm_info
    
    def _detect_metal(self):
        """检测Apple Metal支持"""
        metal_info = {"可用": False}
        try:
            metal_info["可用"] = (
                self.system_info["操作系统"] == "Darwin" and 
                torch.backends.mps.is_available()
            )
            if metal_info["可用"]:
                metal_info["设备"] = "Apple Silicon GPU"
        except:
            pass
            
        self.acceleration_methods["Metal"] = metal_info
    
    def _detect_opencl(self):
        """检测OpenCL支持"""
        opencl_info = {"可用": False}
        try:
            cv2.ocl.setUseOpenCL(True)
            opencl_info["可用"] = cv2.ocl.haveOpenCL()
            if opencl_info["可用"]:
                opencl_info["设备列表"] = []
                try:
                    for device in cv2.ocl.Device_getAll():
                        opencl_info["设备列表"].append({
                            "名称": device.name(),
                            "版本": device.OpenCLVersion(),
                            "类型": self._get_opencl_device_type(device.type())
                        })
                except:
                    pass
        except:
            pass
            
        self.acceleration_methods["OpenCL"] = opencl_info
    
    def _detect_intel_oneapi(self):
        """检测Intel oneAPI支持"""
        intel_info = {"可用": False}
        try:
            import dpctl
            intel_info["可用"] = True
            intel_info["版本"] = dpctl.__version__
            intel_info["设备列表"] = []
            
            try:
                for device in dpctl.get_devices():
                    intel_info["设备列表"].append({
                        "名称": device.name,
                        "驱动版本": device.driver_version,
                        "类型": str(device.device_type)
                    })
            except:
                pass
        except ImportError:
            pass
            
        self.acceleration_methods["Intel oneAPI"] = intel_info
    
    def _detect_opencv_acceleration(self):
        """检测OpenCV内置加速支持"""
        opencv_info = {}
        
        # 检测SIMD指令集
        simd_info = {}
        try:
            cpu_features = cv2.getCPUFeaturesLine()
            for instruction_set in ["SSE", "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2", "AVX", "AVX2", "NEON"]:
                simd_info[instruction_set] = instruction_set in cpu_features
        except:
            for instruction_set in ["SSE", "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2", "AVX", "AVX2", "NEON"]:
                simd_info[instruction_set] = False
        
        opencv_info["SIMD"] = simd_info
        
        self.acceleration_methods["OpenCV内置加速"] = opencv_info
    
    def _detect_numba(self):
        """检测Numba支持"""
        numba_info = {"可用": False}
        try:
            import numba
            numba_info["可用"] = True
            numba_info["版本"] = numba.__version__
            
            # 检测Numba CUDA支持
            try:
                from numba import cuda
                numba_info["cuda支持"] = cuda.is_available()
                if numba_info["cuda支持"]:
                    numba_info["cuda设备数量"] = len(cuda.gpus)
            except:
                numba_info["cuda支持"] = False
        except ImportError:
            pass
            
        self.acceleration_methods["Numba"] = numba_info
    
    def _detect_tensorflow_gpu(self):
        """检测TensorFlow GPU支持"""
        tf_info = {"可用": False}
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            tf_info["可用"] = bool(gpu_devices)
            if tf_info["可用"]:
                tf_info["设备列表"] = [f"{gpu.name} - {gpu.device_type}" for gpu in gpu_devices]
        except:
            pass
            
        self.acceleration_methods["TensorFlow GPU"] = tf_info
    
    def _detect_cupy(self):
        """检测CuPy支持"""
        cupy_info = {"可用": False}
        if CUPY_AVAILABLE:
            try:
                cupy_info["可用"] = True
                cupy_info["版本"] = cp.__version__
                cupy_info["CUDA设备"] = [
                    f"设备{i}: {cp.cuda.runtime.getDeviceProperties(i)['name']}"
                    for i in range(cp.cuda.runtime.getDeviceCount())
                ]
            except:
                pass
        else:
            cupy_info["安装建议"] = "请使用 pip install cupy-cuda11x 安装(替换11x为您的CUDA版本)"
            
        self.acceleration_methods["CuPy"] = cupy_info
    
    def _detect_other_acceleration_libs(self):
        """检测其他加速库支持"""
        other_info = {}
        
        # Dask
        try:
            import dask
            other_info["Dask版本"] = dask.__version__
        except ImportError:
            other_info["Dask"] = "未安装"
        
        # Cython
        try:
            import Cython
            other_info["Cython版本"] = Cython.__version__
        except ImportError:
            other_info["Cython"] = "未安装"
        
        # MPI
        try:
            import mpi4py
            other_info["MPI支持"] = "通过mpi4py可用"
        except ImportError:
            other_info["MPI支持"] = "未安装"
        
        # Ray
        try:
            import ray
            other_info["Ray版本"] = ray.__version__
        except ImportError:
            other_info["Ray"] = "未安装"
            
        self.acceleration_methods["其他加速库"] = other_info
    
    def _get_opencl_device_type(self, device_type: int) -> str:
        """转换OpenCL设备类型为可读字符串"""
        type_map = {
            0x1: "默认",
            0x2: "CPU",
            0x4: "GPU",
            0x8: "加速器",
            0x10: "自定义",
            0xFFFFFFFF: "全部"
        }
        return type_map.get(device_type, "未知")
    
    def _generate_optimization_advice(self) -> List[str]:
        """生成优化建议"""
        advice = []
        methods = self.acceleration_methods
        
        # CUDA建议
        if methods["CUDA"]["可用"]:
            advice.append(
                "✅ 检测到NVIDIA GPU，推荐使用CUDA加速: "
                "安装PyTorch CUDA版本 (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)"
            )
            if methods["CUDA"].get("opencv_cuda", False):
                advice.append("✅ OpenCV已启用CUDA支持，可以使用cv2.cuda模块进行加速")
            else:
                advice.append(
                    "⚠️ OpenCV未启用CUDA支持，建议重新安装OpenCV: "
                    "`pip install opencv-python-headless opencv-contrib-python-headless`"
                )
        
        # ROCm建议
        if methods["ROCm"]["可用"]:
            advice.append(
                "✅ 检测到AMD GPU，推荐使用ROCm加速: "
                "安装PyTorch ROCm版本 (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6`)"
            )
        
        # Metal建议
        if methods["Metal"]["可用"]:
            advice.append(
                "✅ 检测到Apple Silicon GPU，推荐使用Metal加速: "
                "在PyTorch中使用`device = torch.device('mps')`"
            )
        
        # OpenCL建议
        if methods["OpenCL"]["可用"]:
            device_list = methods["OpenCL"].get("设备列表", [])
            gpu_devices = [d for d in device_list if "GPU" in d["类型"]]
            
            if gpu_devices:
                advice.append(
                    "✅ 检测到OpenCL GPU设备，可以使用OpenCL加速: "
                    "在OpenCV中启用`cv2.ocl.setUseOpenCL(True)`"
                )
            else:
                advice.append("ℹ️ 检测到OpenCL CPU设备，可以使用OpenCL进行CPU加速")
        
        # Intel oneAPI建议
        if methods["Intel oneAPI"]["可用"]:
            advice.append(
                "✅ 检测到Intel oneAPI支持，可以使用Intel GPU加速: "
                "安装Intel扩展 (`pip install dpctl`)"
            )
        
        # Numba建议
        if methods["Numba"]["可用"]:
            advice.append(
                "✅ 检测到Numba，可以使用JIT编译加速Python代码: "
                "使用`@numba.jit`装饰器优化关键函数"
            )
            if methods["Numba"].get("cuda支持", False):
                advice.append("✅ Numba支持CUDA，可以使用`@numba.cuda.jit`加速GPU计算")
        
        # TensorFlow建议
        if methods["TensorFlow GPU"]["可用"]:
            advice.append("✅ TensorFlow已检测到GPU设备，可以使用GPU加速计算")
        
        # CuPy建议
        if methods["CuPy"]["可用"]:
            advice.append("✅ CuPy已安装，可以使用CuPy加速NumPy兼容的GPU计算")
        else:
            advice.append("ℹ️ 未安装CuPy，如需GPU加速NumPy操作，可以安装: `pip install cupy-cuda11x`")
        
        # OpenCV SIMD建议
        opencv_accel = methods.get("OpenCV内置加速", {})
        active_simd = [k for k, v in opencv_accel.get("SIMD", {}).items() if v]
        if active_simd:
            advice.append(f"✅ OpenCV已启用CPU SIMD指令集: {', '.join(active_simd)}")
        
        # 如果没有检测到任何GPU加速
        if not any([
            methods["CUDA"]["可用"],
            methods["ROCm"]["可用"],
            methods["Metal"]["可用"],
            methods["OpenCL"]["可用"],
            methods["Intel oneAPI"]["可用"],
            methods["TensorFlow GPU"]["可用"]
        ]):
            advice.extend([
                "⚠️ 未检测到GPU加速支持，可以使用以下CPU优化方案:",
                "1. 使用OpenCV的SIMD优化 (已自动启用)",
                "2. 安装Numba进行JIT编译 (`pip install numba`)",
                "3. 使用多线程/多进程处理",
                "4. 优化算法复杂度或降低分辨率"
            ])
        
        return advice
    
    def print_report(self):
        """打印检测报告"""
        report = []
        report.append("="*80)
        report.append(f"GPU与加速支持综合检测报告 (v{__version__})")
        report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        # 系统信息
        report.append("\n[系统信息]")
        for key, value in self.system_info.items():
            report.append(f"{key}: {value}")
        
        # 加速支持情况
        report.append("\n[加速支持]")
        for method_name, method_info in self.acceleration_methods.items():
            report.append(f"\n{method_name}:")
            if isinstance(method_info, dict):
                for key, value in method_info.items():
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            report.append(f"    {sub_key}: {sub_value}")
                    else:
                        report.append(f"  {key}: {value}")
            else:
                report.append(f"  {method_info}")
        
        # 优化建议
        report.append("\n[优化建议]")
        for i, advice in enumerate(self._generate_optimization_advice(), 1):
            report.append(f"{i}. {advice}")
        
        # 安装命令
        report.append("\n[推荐安装命令]")
        report.append("-"*30)
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
            "Tabulate": "pip install tabulate"
        }
        for lib, cmd in commands.items():
            report.append(f"{lib}: {cmd}")
        
        # 最终报告
        full_report = "\n".join(report)
        print(full_report)
        
        # 保存报告到文件
        with open(self.report_filename, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n报告已保存到: {self.report_filename}")

if __name__ == "__main__":
    detector = GPUAccelerationDetector()
    detector.detect_all()
    detector.print_report()