# LABWORK 3 - 1D KERNEL

import numba
from numba import cuda
from numba.cuda.cudadrv import enums
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
from numba import vectorize

Data_path = "/content/drive/MyDrive/HPC/"
Image_path = Data_path+"Test_org.jpg"

# blockDim.x,y,z gives the number of threads in a block, in the particular direction
# gridDim.x,y,z gives the number of blocks in a grid, in the particular direction
# blockDim.x * gridDim.x gives the number of threads in a grid

# @vectorize(['float32(float32, float32)'], target='cuda')


@cuda.jit
def grayscale(src, dst):
    # where are we in the input?
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    print(tidx)
    g = np.uint16(
        (int(src[tidx, 0]) + int(src[tidx, 1]) + int(src[tidx, 2])) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g


# Image data
img_data = plt.imread(Image_path)

# Shape of the figure
(imageHeight, imageWidth, _) = img_data.shape
pixelCount = imageWidth * imageHeight


def compare(blockSize, output=False):
    # Grid size -> chunk
    # int to ensure it's an interger
    gridSize = int(pixelCount / blockSize)

    # Flatten image into an array
    img_flatten = img_data.flatten().reshape((pixelCount, 3))
    # print(img_flatten)

    # Start timing
    start_time = time.time()

    # Copy image to the device from host(CPU)
    devSrc = cuda.to_device(img_flatten)

    # Allocate memory on the device (GPU)
    devDst = cuda.device_array((pixelCount, 3), np.uint16)

    grayscale[gridSize, blockSize](devSrc, devDst)

    cuda.synchronize()

    # Copy from device to host
    hostDst = devDst.copy_to_host()

    # Stop timing
    end_time = time.time()

    # Get the running time
    run_time = end_time-start_time

    if output == True:
        return run_time, hostDst
    else:
        return run_time

# Output run_time and hostDst


output = False

if output == True:
    blockSize = 2
    run_time, hostDst = compare(blockSize, output=True)
    # Reshape the image back to the original form
    res_image = np.reshape(hostDst, (imageHeight, imageWidth, 3))[:, :, 0]

    # Show the resule image
    plt.imshow(res_image)

    # Save the image
    plt.savefig(Data_path+"Test_result_LW4.jpg")
    print("The run time is", run_time, "s")

else:
    runtime_list = []
    blockSize_final = []
    for i in range(6):
        blockSize = 2**i
        run_time = compare(blockSize, output=False)
        runtime_list.append(run_time)
        blockSize_final.append(blockSize)
        print(f"The runtime for the blockSize {blockSize} is {run_time}")

plt.plot(runtime_list)
plt.xlabel("")
plt.title("BlockSize vs Runtime comparision")
