import cv2
import torch
import platform
import numpy as np
from typing import Dict, List, Optional

class 加速检测器:
    def __init__(self):
        self.系统信息 = {
            "操作系统": platform.system(),
            "处理器": platform.processor(),
            "Python版本": platform.python_version(),
            "OpenCV版本": cv2.__version__,
            "PyTorch版本": torch.__version__
        }
        self.可用加速方案 = {}
        
    def 检测所有(self) -> Dict[str, dict]:
        """检查所有可用的加速方案"""
        self._检测_cuda()
        self._检测_rocm()
        self._检测_metal()
        self._检测_opencl()
        self._检测_intel_oneapi()
        self._检测_opencv加速()
        self._检测_numba()
        
        return {
            "系统信息": self.系统信息,
            "加速方案": self.可用加速方案,
            "优化建议": self._生成优化建议()
        }
    
    def _检测_cuda(self):
        """检查NVIDIA CUDA支持"""
        cuda信息 = {"可用": False}
        try:
            if torch.cuda.is_available():
                cuda信息["可用"] = True
                cuda信息["设备数量"] = torch.cuda.device_count()
                cuda信息["设备列表"] = []
                
                for i in range(torch.cuda.device_count()):
                    设备信息 = {
                        "名称": torch.cuda.get_device_name(i),
                        "计算能力": f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}",
                        "显存": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                    }
                    cuda信息["设备列表"].append(设备信息)
                    
                # 检查OpenCV CUDA支持
                try:
                    cuda信息["opencv_cuda"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
                except:
                    cuda信息["opencv_cuda"] = False
        except:
            pass
        
        self.可用加速方案["CUDA"] = cuda信息
    
    def _检测_rocm(self):
        """检查AMD ROCm支持"""
        rocm信息 = {"可用": False}
        try:
            rocm信息["可用"] = torch.version.hip is not None
            if rocm信息["可用"]:
                rocm信息["版本"] = torch.version.hip
        except:
            pass
            
        self.可用加速方案["ROCm"] = rocm信息
    
    def _检测_metal(self):
        """检查Apple Metal支持"""
        metal信息 = {"可用": False}
        try:
            metal信息["可用"] = (
                self.系统信息["操作系统"] == "Darwin" and 
                torch.backends.mps.is_available()
            )
            if metal信息["可用"]:
                metal信息["设备"] = "Apple Silicon GPU"
        except:
            pass
            
        self.可用加速方案["Metal"] = metal信息
    
    def _检测_opencl(self):
        """检查OpenCL支持"""
        opencl信息 = {"可用": False}
        try:
            cv2.ocl.setUseOpenCL(True)
            opencl信息["可用"] = cv2.ocl.haveOpenCL()
            if opencl信息["可用"]:
                opencl信息["设备列表"] = []
                try:
                    for device in cv2.ocl.Device_getAll():
                        opencl信息["设备列表"].append({
                            "名称": device.name(),
                            "版本": device.OpenCLVersion(),
                            "类型": self._获取_opencl_设备类型(device.type())
                        })
                except:
                    pass
        except:
            pass
            
        self.可用加速方案["OpenCL"] = opencl信息
    
    def _检测_intel_oneapi(self):
        """检查Intel oneAPI支持"""
        intel信息 = {"可用": False}
        try:
            import dpctl
            intel信息["可用"] = True
            intel信息["版本"] = dpctl.__version__
            intel信息["设备列表"] = []
            
            try:
                for device in dpctl.get_devices():
                    intel信息["设备列表"].append({
                        "名称": device.name,
                        "驱动版本": device.driver_version,
                        "类型": str(device.device_type)
                    })
            except:
                pass
        except ImportError:
            pass
            
        self.可用加速方案["Intel oneAPI"] = intel信息
    
    def _检测_opencv加速(self):
        """检查OpenCV内置加速支持"""
        opencv信息 = {}
        
        # 检查SIMD指令集
        simd信息 = {}
        try:
            cpu特性 = cv2.getCPUFeaturesLine()
            for 指令集 in ["SSE", "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2", "AVX", "AVX2", "NEON"]:
                simd信息[指令集] = 指令集 in cpu特性
        except:
            # 如果getCPUFeaturesLine不可用，全部设为False
            for 指令集 in ["SSE", "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2", "AVX", "AVX2", "NEON"]:
                simd信息[指令集] = False
        
        opencv信息["SIMD"] = simd信息
        
        self.可用加速方案["OpenCV内置加速"] = opencv信息
    
    def _检测_numba(self):
        """检查Numba支持"""
        numba信息 = {"可用": False}
        try:
            import numba
            numba信息["可用"] = True
            numba信息["版本"] = numba.__version__
            
            # 检查Numba CUDA支持
            try:
                from numba import cuda
                numba信息["cuda支持"] = cuda.is_available()
                if numba信息["cuda支持"]:
                    numba信息["cuda设备数量"] = len(cuda.gpus)
            except:
                numba信息["cuda支持"] = False
        except ImportError:
            pass
            
        self.可用加速方案["Numba"] = numba信息
    
    def _获取_opencl_设备类型(self, device_type: int) -> str:
        """转换OpenCL设备类型为可读字符串"""
        类型映射 = {
            0x1: "默认",
            0x2: "CPU",
            0x4: "GPU",
            0x8: "加速器",
            0x10: "自定义",
            0xFFFFFFFF: "全部"
        }
        return 类型映射.get(device_type, "未知")
    
    def _生成优化建议(self) -> List[str]:
        """生成优化建议"""
        建议列表 = []
        加速方案 = self.可用加速方案
        
        # 根据检测结果生成建议
        if 加速方案["CUDA"]["可用"]:
            建议列表.append(
                "✅ 检测到NVIDIA GPU，推荐使用CUDA加速: "
                f"安装PyTorch CUDA版本 (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)"
            )
            if 加速方案["CUDA"]["opencv_cuda"]:
                建议列表.append(
                    "✅ OpenCV已启用CUDA支持，可以使用cv2.cuda模块进行加速"
                )
            else:
                建议列表.append(
                    "⚠️ OpenCV未启用CUDA支持，建议重新安装OpenCV: "
                    "`pip install opencv-python-headless opencv-contrib-python-headless`"
                )
        
        if 加速方案["ROCm"]["可用"]:
            建议列表.append(
                "✅ 检测到AMD GPU，推荐使用ROCm加速: "
                "安装PyTorch ROCm版本 (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6`)"
            )
        
        if 加速方案["Metal"]["可用"]:
            建议列表.append(
                "✅ 检测到Apple Silicon GPU，推荐使用Metal加速: "
                "在PyTorch中使用`device = torch.device('mps')`"
            )
        
        if 加速方案["OpenCL"]["可用"]:
            设备列表 = 加速方案["OpenCL"].get("设备列表", [])
            gpu设备 = [d for d in 设备列表 if "GPU" in d["类型"]]
            
            if gpu设备:
                建议列表.append(
                    "✅ 检测到OpenCL GPU设备，可以使用OpenCL加速: "
                    "在OpenCV中启用`cv2.ocl.setUseOpenCL(True)`"
                )
            else:
                建议列表.append(
                    "ℹ️ 检测到OpenCL CPU设备，可以使用OpenCL进行CPU加速"
                )
        
        if 加速方案["Intel oneAPI"]["可用"]:
            建议列表.append(
                "✅ 检测到Intel oneAPI支持，可以使用Intel GPU加速: "
                "安装Intel扩展 (`pip install dpctl`)"
            )
        
        if 加速方案["Numba"]["可用"]:
            建议列表.append(
                "✅ 检测到Numba，可以使用JIT编译加速Python代码: "
                "使用`@numba.jit`装饰器优化关键函数"
            )
            if 加速方案["Numba"].get("cuda支持", False):
                建议列表.append(
                    "✅ Numba支持CUDA，可以使用`@numba.cuda.jit`加速GPU计算"
                )
        
        # 检查OpenCV优化选项
        opencv加速 = 加速方案.get("OpenCV内置加速", {})
        激活的simd = [k for k, v in opencv加速.get("SIMD", {}).items() if v]
        if 激活的simd:
            建议列表.append(
                f"✅ OpenCV已启用CPU SIMD指令集: {', '.join(激活的simd)}"
            )
        
        # 如果没有检测到任何GPU加速
        if not any([
            加速方案["CUDA"]["可用"],
            加速方案["ROCm"]["可用"],
            加速方案["Metal"]["可用"],
            加速方案["OpenCL"]["可用"],
            加速方案["Intel oneAPI"]["可用"]
        ]):
            建议列表.extend([
                "⚠️ 未检测到GPU加速支持，可以使用以下CPU优化方案:",
                "1. 使用OpenCV的SIMD优化 (已自动启用)",
                "2. 安装Numba进行JIT编译 (`pip install numba`)",
                "3. 使用多线程/多进程处理",
                "4. 优化算法复杂度或降低分辨率"
            ])
        
        return 建议列表
    
    def 打印报告(self):
        """打印检测报告"""
        print("="*50)
        print("系统加速能力检测报告")
        print("="*50)
        
        # 打印系统信息
        print("\n[系统信息]")
        for 键, 值 in self.系统信息.items():
            print(f"{键}: {值}")
        
        # 打印加速支持情况
        print("\n[加速支持]")
        for 方案名称, 方案信息 in self.可用加速方案.items():
            print(f"\n{方案名称}:")
            if isinstance(方案信息, dict):
                for 键, 值 in 方案信息.items():
                    if isinstance(值, dict):
                        print(f"  {键}:")
                        for 子键, 子值 in 值.items():
                            print(f"    {子键}: {子值}")
                    else:
                        print(f"  {键}: {值}")
            else:
                print(f"  {方案信息}")
        
        # 打印优化建议
        print("\n[优化建议]")
        for 序号, 建议 in enumerate(self._生成优化建议(), 1):
            print(f"{序号}. {建议}")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    检测器 = 加速检测器()
    检测器.检测所有()
    检测器.打印报告()
