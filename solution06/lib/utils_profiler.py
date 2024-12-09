import os
import pstats
from typing import Optional, List, Union

import psutil
import pynvml  # pip install nvidia-ml-py


def decode_maybe(in_str: Union[str, bytes]):
    try:
        return str(in_str.decode("utf8"))
    except AttributeError:
        return str(in_str)


class GPUProfiler:
    """
    Installation:
        pip install nvidia-ml-py3 pynvml
        If the packages are not found, try to install them from nvidias package index:
        pip install nvidia-ml-py3 pynvml --extra-index-url "\nhttps://pypi.ngc.nvidia.com"

    Notes:
        Explanation for "Utilization GPU (Util)" and "Utilization GPU Memory (UMem)":
        nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t * utilization)
        nvmlUtilization_t Struct Reference
        unsigned int gpu
            Percent of time over the past second in which any work has been executing on the GPU.
        unsigned int memory
            Percent of time over the past second in which any framebuffer memory
            has been read or stored.

    Usage:
        >>> gpu_profiler = GPUProfiler()
        >>> print(gpu_profiler.profile_to_str())
    """

    def __init__(self):
        self.gpu_count = -1
        self.gpu_handles = []
        pynvml.nvmlInit()
        self.gpu_count = pynvml.nvmlDeviceGetCount()
        if self.gpu_count == 0:
            self.gpu_count = 1
        self.gpu_handles = []
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            except Exception as e:
                print(
                    f"WARNING: Could not get handle for GPU {i}: "
                    f"{e} ({e.__class__.__name__})"
                )
                handle = None
            self.gpu_handles.append(handle)

    def get_gpu_numbers(self):
        """
        Try to determine gpu numbers from env variables, fallback is to profile all gpus

        Returns:
            List with GPU numbers
        """
        gpu_numbers = list(range(self.gpu_count))
        device_env = os.getenv("CUDA_VISIBLE_DEVICES")
        if device_env is not None:
            try:
                gpu_numbers = [
                    int(x.strip()) for x in device_env.split(",") if x.strip() != ""
                ]
            except Exception:
                pass
            if len(gpu_numbers) == 0:
                gpu_numbers = list(range(self.gpu_count))
        return gpu_numbers

    def profile_gpu(self, gpu_numbers: Optional[List[int]] = None):
        """
        Profile GPU utilization

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            Lists with entries for each GPU with content:
                name, total memory (GB), used memory (GB), gpu load (0-1), memory load (0-1), temperature (°C)

        """
        gpu_numbers = self.get_gpu_numbers() if gpu_numbers is None else gpu_numbers
        gpu_numbers = [i for i in gpu_numbers if self.gpu_handles[i] is not None]

        mem_objs = [
            pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handles[i]) for i in gpu_numbers
        ]
        mem_total = [mem_obj.total / 1024 ** 3 for mem_obj in mem_objs]
        mem_used = [mem_obj.used / 1024 ** 3 for mem_obj in mem_objs]
        names = [
            decode_maybe(pynvml.nvmlDeviceGetName(self.gpu_handles[i]))
            for i in gpu_numbers
        ]
        load_objs = [
            pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handles[i])
            for i in gpu_numbers
        ]
        load_gpu = [load_obj.gpu / 100 for load_obj in load_objs]
        load_gpu_mem = [load_obj.memory / 100 for load_obj in load_objs]
        temp = [
            pynvml.nvmlDeviceGetTemperature(
                self.gpu_handles[i], pynvml.NVML_TEMPERATURE_GPU
            )
            for i in gpu_numbers
        ]
        return names, mem_total, mem_used, load_gpu, load_gpu_mem, temp

    def profile_to_str(self, gpu_numbers: Optional[List[int]] = None):
        """
        Use profile output to create a string

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            profile string for output
        """
        names, mem_total, mem_used, load_gpu, load_gpu_mem, temp = self.profile_gpu(
            gpu_numbers
        )
        ram_total, ram_used = profile_ram()
        # average / sum over all GPUs
        sum_mem_total: float = sum(mem_total)
        sum_mem_used: float = sum(mem_used)
        # gpu_mem_percent: float = gpu_mem_used / gpu_mem_total
        avg_load_gpu: float = sum(load_gpu) / max(1, len(load_gpu))
        avg_load_gpu_mem: float = sum(load_gpu_mem) / max(1, len(load_gpu_mem))
        avg_temp: float = sum(temp) / max(1, len(temp))

        # log the values
        gpu_names_str = " ".join(set(names))
        multi_load, multi_load_mem, multi_temp, multi_mem = "", "", "", ""
        if len(load_gpu) > 1:
            multi_load = " " + ", ".join([f"{ld:.0%}" for ld in load_gpu])
            multi_load_mem = " " + ", ".join([f"{ld:.0%}" for ld in load_gpu_mem])
            multi_temp = " " + ", ".join([f"{ld:d}" for ld in temp])
            multi_mem = " " + ", ".join([f"{mem:.1f}GB" for mem in mem_used])

        out_str = (
            f"RAM {ram_used:.1f}/{ram_total:.1f} "
            f"GPU {gpu_names_str} Util {avg_load_gpu:.0%}{multi_load} "
            f"UMem {avg_load_gpu_mem:.0%}{multi_load_mem} "
            f"Mem {sum_mem_used:.1f}/{sum_mem_total:.1f}{multi_mem} "
            f"Temp {avg_temp:.0f}°C{multi_temp}"
        )

        return out_str

    def check_gpus_have_errors(self, gpu_numbers: Optional[List[int]] = None):
        """
        Check for obscure errors that are hard to find otherwise.

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            profile string for output
        """
        gpu_numbers = self.get_gpu_numbers() if gpu_numbers is None else gpu_numbers
        has_errors = False
        for i in gpu_numbers:
            if self.gpu_handles[i] is None:
                has_errors = True
                print(f"ERROR: GPU {i} handle could not be created at startup.")
                continue
            try:
                pynvml.nvmlDeviceGetClockInfo(
                    self.gpu_handles[i], pynvml.NVML_CLOCK_GRAPHICS
                )
            except Exception as e:
                has_errors = True
                print(f"ERROR: GPU {i} has error: {e} ({type(e)}")
        return has_errors


def profile_ram():
    """

    Returns:
        RAM total (GB), RAM used (GB)
    """
    mem = psutil.virtual_memory()
    ram_total: float = mem.total / 1024 ** 3
    ram_used: float = mem.used / 1024 ** 3
    return ram_total, ram_used


def read_profile_output(filename):
    """
    Read the output of cProfile module and print a report.
    """
    stats = pstats.Stats(filename)
    stats.strip_dirs()
    stats.sort_stats("time", "calls")
    stats.print_stats(20)

    stats.sort_stats("cumulative", "calls")
    stats.print_stats(20)
