import numba
from numba import cuda
from numba.cuda.cudadrv import enums
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
from numba import vectorize

#LABWORK4 - 2D KERNEL

Data_path="/content/drive/MyDrive/HPC/"
Image_path=Data_path+"Test_org.jpg"

# blockDim.x,y,z gives the number of threads in a block, in the particular direction
# gridDim.x,y,z gives the number of blocks in a grid, in the particular direction
# blockDim.x * gridDim.x gives the number of threads in a grid

@cuda.jit
def grayscale(src, dst):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
  g = np.uint16(src[tidx,tidy, 0] + src[tidx,tidy, 1] + src[tidx,tidy, 2]) / 3
  dst[tidx,tidy, 0] = dst[tidx,tidy, 1] = dst[tidx,tidy, 2] = g


# Image data
img_data=plt.imread(Image_path)

# Shape of the figure
(imageHeight,imageWidth,_)=img_data.shape
pixelCount = imageWidth * imageHeight
gray_img = np.array(img_data, copy=True)

def compare(blockSize,output=False):
  #Grid size -> chunk
  #int to ensure it's an interger
  # BlockSize should be the multiplication of 32
  grid_1 = int(imageHeight/blockSize)
  grid_2 = int(imageWidth/blockSize)
  print(grid_1,grid_2)
  gridSize=(grid_1,grid_2)

  blockSize=(blockSize,blockSize)

  # Start timing
  start_time=time.time()

  # Copy image to the device from host(CPU)
  devSrc = cuda.to_device(img_data)

  # Allocate memory on the device (GPU)
  devDst = cuda.device_array((imageHeight,imageWidth,3), np.uint16)

  grayscale[gridSize,blockSize](devSrc, devDst)

  #Copy from device to host
  hostDst = devDst.copy_to_host()

  # Stop timing
  end_time=time.time()

  #Get the running time
  run_time=end_time-start_time

  if output == True:
    return run_time, hostDst
  else: return run_time
  
#Output run_time and hostDst
output=False

if output == True:
  blockSize = 4
  run_time,hostDst= compare(blockSize,output=True)

  # Show the resule image
  plt.imshow(hostDst)

  # Save the image
  plt.savefig(Data_path+"Test_result_LW4.jpg")
  print("The run time is",run_time,"s")

else:
  runtime_list=[]
  blockSize_final=[]
  for i in range(6):
    blockSize=2**i
    run_time=compare(blockSize,output=False)
    runtime_list.append(run_time)
    blockSize_final.append(blockSize)
    print(f"The runtime for the blockSize {blockSize} is {run_time}")

  name=f'LW5 - BlockSize vs Runtime comparision' # - Shared memory')
  plt.plot(blockSize_final,runtime_list)
  plt.xticks(blockSize_final,blockSize_label)
  plt.title("BlockSize vs Runtime comparision") #- Shared memory")
  plt.savefig(Data_path + name + ",png")