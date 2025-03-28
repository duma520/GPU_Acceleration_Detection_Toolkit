import cv2
import torch
import platform
import numpy as np
from typing import Dict, List, Optional

class AccelerationChecker:
    def __init__(self):
        self.system_info = {
            "System": platform.system(),
            "Processor": platform.processor(),
            "Python": platform.python_version(),
            "OpenCV": cv2.__version__,
            "PyTorch": torch.__version__
        }
        self.available_accelerations = {}
        
    def check_all(self) -> Dict[str, dict]:
        """检查所有可用的加速方案"""
        self._check_cuda()
        self._check_rocm()
        self._check_metal()
        self._check_opencl()
        self._check_intel_oneapi()
        self._check_opencv_acceleration()
        self._check_numba()
        
        return {
            "system_info": self.system_info,
            "accelerations": self.available_accelerations,
            "recommendations": self._generate_recommendations()
        }
    
    def _check_cuda(self):
        """检查NVIDIA CUDA支持"""
        cuda_info = {"available": False}
        try:
            if torch.cuda.is_available():
                cuda_info["available"] = True
                cuda_info["device_count"] = torch.cuda.device_count()
                cuda_info["devices"] = []
                
                for i in range(torch.cuda.device_count()):
                    device_info = {
                        "name": torch.cuda.get_device_name(i),
                        "capability": f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}",
                        "memory": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                    }
                    cuda_info["devices"].append(device_info)
                    
                # 检查OpenCV CUDA支持
                try:
                    cuda_info["opencv_cuda"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
                except:
                    cuda_info["opencv_cuda"] = False
        except:
            pass
        
        self.available_accelerations["CUDA"] = cuda_info
    
    def _check_rocm(self):
        """检查AMD ROCm支持"""
        rocm_info = {"available": False}
        try:
            rocm_info["available"] = torch.version.hip is not None
            if rocm_info["available"]:
                rocm_info["version"] = torch.version.hip
        except:
            pass
            
        self.available_accelerations["ROCm"] = rocm_info
    
    def _check_metal(self):
        """检查Apple Metal支持"""
        metal_info = {"available": False}
        try:
            metal_info["available"] = (
                self.system_info["System"] == "Darwin" and 
                torch.backends.mps.is_available()
            )
            if metal_info["available"]:
                metal_info["device"] = "Apple Silicon GPU"
        except:
            pass
            
        self.available_accelerations["Metal"] = metal_info
    
    def _check_opencl(self):
        """检查OpenCL支持"""
        opencl_info = {"available": False}
        try:
            cv2.ocl.setUseOpenCL(True)
            opencl_info["available"] = cv2.ocl.haveOpenCL()
            if opencl_info["available"]:
                opencl_info["devices"] = []
                try:
                    for device in cv2.ocl.Device_getAll():
                        opencl_info["devices"].append({
                            "name": device.name(),
                            "version": device.OpenCLVersion(),
                            "type": self._get_opencl_device_type(device.type())
                        })
                except:
                    pass
        except:
            pass
            
        self.available_accelerations["OpenCL"] = opencl_info
    
    def _check_intel_oneapi(self):
        """检查Intel oneAPI支持"""
        intel_info = {"available": False}
        try:
            import dpctl
            intel_info["available"] = True
            intel_info["version"] = dpctl.__version__
            intel_info["devices"] = []
            
            try:
                for device in dpctl.get_devices():
                    intel_info["devices"].append({
                        "name": device.name,
                        "driver_version": device.driver_version,
                        "type": str(device.device_type)
                    })
            except:
                pass
        except ImportError:
            pass
            
        self.available_accelerations["Intel oneAPI"] = intel_info
    
    def _check_opencv_acceleration(self):
        """检查OpenCV内置加速支持"""
        opencv_info = {}
        
        # 检查SIMD指令集
        simd_info = {}
        try:
            cpu_features = cv2.getCPUFeaturesLine()
            for feature in ["SSE", "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2", "AVX", "AVX2", "NEON"]:
                simd_info[feature] = feature in cpu_features
        except:
            # 如果getCPUFeaturesLine不可用，全部设为False
            for feature in ["SSE", "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2", "AVX", "AVX2", "NEON"]:
                simd_info[feature] = False
        
        opencv_info["SIMD"] = simd_info
        
        self.available_accelerations["OpenCV Built-in"] = opencv_info
    
    def _check_numba(self):
        """检查Numba支持"""
        numba_info = {"available": False}
        try:
            import numba
            numba_info["available"] = True
            numba_info["version"] = numba.__version__
            
            # 检查Numba CUDA支持
            try:
                from numba import cuda
                numba_info["cuda_support"] = cuda.is_available()
                if numba_info["cuda_support"]:
                    numba_info["cuda_devices"] = len(cuda.gpus)
            except:
                numba_info["cuda_support"] = False
        except ImportError:
            pass
            
        self.available_accelerations["Numba"] = numba_info
    
    def _get_opencl_device_type(self, device_type: int) -> str:
        """转换OpenCL设备类型为可读字符串"""
        types = {
            0x1: "DEFAULT",
            0x2: "CPU",
            0x4: "GPU",
            0x8: "Accelerator",
            0x10: "Custom",
            0xFFFFFFFF: "All"
        }
        return types.get(device_type, "UNKNOWN")
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        acc = self.available_accelerations
        
        # 根据检测结果生成建议
        if acc["CUDA"]["available"]:
            recommendations.append(
                "✅ 检测到NVIDIA GPU，推荐使用CUDA加速: "
                f"安装PyTorch CUDA版本 (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)"
            )
            if acc["CUDA"]["opencv_cuda"]:
                recommendations.append(
                    "✅ OpenCV已启用CUDA支持，可以使用cv2.cuda模块进行加速"
                )
            else:
                recommendations.append(
                    "⚠️ OpenCV未启用CUDA支持，建议重新安装OpenCV: "
                    "`pip install opencv-python-headless opencv-contrib-python-headless`"
                )
        
        if acc["ROCm"]["available"]:
            recommendations.append(
                "✅ 检测到AMD GPU，推荐使用ROCm加速: "
                "安装PyTorch ROCm版本 (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6`)"
            )
        
        if acc["Metal"]["available"]:
            recommendations.append(
                "✅ 检测到Apple Silicon GPU，推荐使用Metal加速: "
                "在PyTorch中使用`device = torch.device('mps')`"
            )
        
        if acc["OpenCL"]["available"]:
            devices = acc["OpenCL"].get("devices", [])
            gpu_devices = [d for d in devices if "GPU" in d["type"]]
            
            if gpu_devices:
                recommendations.append(
                    "✅ 检测到OpenCL GPU设备，可以使用OpenCL加速: "
                    "在OpenCV中启用`cv2.ocl.setUseOpenCL(True)`"
                )
            else:
                recommendations.append(
                    "ℹ️ 检测到OpenCL CPU设备，可以使用OpenCL进行CPU加速"
                )
        
        if acc["Intel oneAPI"]["available"]:
            recommendations.append(
                "✅ 检测到Intel oneAPI支持，可以使用Intel GPU加速: "
                "安装Intel扩展 (`pip install dpctl`)"
            )
        
        if acc["Numba"]["available"]:
            recommendations.append(
                "✅ 检测到Numba，可以使用JIT编译加速Python代码: "
                "使用`@numba.jit`装饰器优化关键函数"
            )
            if acc["Numba"].get("cuda_support", False):
                recommendations.append(
                    "✅ Numba支持CUDA，可以使用`@numba.cuda.jit`加速GPU计算"
                )
        
        # 检查OpenCV优化选项
        opencv_acc = acc.get("OpenCV Built-in", {})
        active_simd = [k for k, v in opencv_acc.get("SIMD", {}).items() if v]
        if active_simd:
            recommendations.append(
                f"✅ OpenCV已启用CPU SIMD指令集: {', '.join(active_simd)}"
            )
        
        # 如果没有检测到任何GPU加速
        if not any([
            acc["CUDA"]["available"],
            acc["ROCm"]["available"],
            acc["Metal"]["available"],
            acc["OpenCL"]["available"],
            acc["Intel oneAPI"]["available"]
        ]):
            recommendations.extend([
                "⚠️ 未检测到GPU加速支持，可以使用以下CPU优化方案:",
                "1. 使用OpenCV的SIMD优化 (已自动启用)",
                "2. 安装Numba进行JIT编译 (`pip install numba`)",
                "3. 使用多线程/多进程处理",
                "4. 优化算法复杂度或降低分辨率"
            ])
        
        return recommendations
    
    def print_report(self):
        """打印检测报告"""
        print("="*50)
        print("系统加速能力检测报告")
        print("="*50)
        
        # 打印系统信息
        print("\n[系统信息]")
        for k, v in self.system_info.items():
            print(f"{k}: {v}")
        
        # 打印加速支持情况
        print("\n[加速支持]")
        for acc_name, acc_info in self.available_accelerations.items():
            print(f"\n{acc_name}:")
            if isinstance(acc_info, dict):
                for k, v in acc_info.items():
                    if isinstance(v, dict):
                        print(f"  {k}:")
                        for sk, sv in v.items():
                            print(f"    {sk}: {sv}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"  {acc_info}")
        
        # 打印优化建议
        print("\n[优化建议]")
        for i, rec in enumerate(self._generate_recommendations(), 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    checker = AccelerationChecker()
    checker.check_all()
    checker.print_report()
