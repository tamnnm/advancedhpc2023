#LABWORK2.py
# Tam Nguyen Ngoc-Minh
# Student ID: D22SA.001
import numba
from numba import cuda
from numba.cuda.cudadrv import enums

# Get the device name and number of devices
print("Detecting available device",cuda.detect())
print("Device found:",cuda.devices)
print(cuda.gpus)
device = cuda.select_device(0)


# This is the fastest way to print out the needed attribute
# special_info="CLOCK_RATE"

# Print all the attribute
attribs= [name.replace("CU_DEVICE_ATTRIBUTE_", "") for name in dir(enums) if name.startswith("CU_DEVICE_ATTRIBUTE_")]
for attr in attribs:
    # if special_info is in attr:
        print(attr, '=', getattr(device, attr))

# Memory info
print(cuda.current_context().get_memory_info())

# Core counts, multiprocessor count
cc_cores_per_SM_dict = {
    (2,0) : 32,
    (2,1) : 48,
    (3,0) : 192,
    (3,5) : 192,
    (3,7) : 192,
    (5,0) : 128,
    (5,2) : 128,
    (6,0) : 64,
    (6,1) : 128,
    (7,0) : 64,
    (7,5) : 64,
    (8,0) : 64,
    (8,6) : 128,
    (8,9) : 128,
    (9,0) : 128
    }
mul_pc = device.MULTIPROCESSOR_COUNT
my_cc = device.compute_capability
cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
total_cores = cores_per_sm*mul_pc
print(f"GPU compute capability: {my_cc}")
print(f"GPU total number of SMs: {mul_pc}")
print(f"total cores: {total_cores}")