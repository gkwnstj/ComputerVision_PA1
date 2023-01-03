import cv2       ##cv2.imread, cv2.imwrite, cv2.cvtColor, cv2.Laplacian, cv2.Sobel, cv2.resize
import numpy as np
##################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity
##################################################################################

#############################       Task_1      ############################################

############################# Load_data ##########################################
ih1 =  cv2.imread('upsampled.png', cv2.IMREAD_GRAYSCALE)   # ih shape :  (256, 256) as initial high resolution(ih0)
ih = ih1.astype(np.float64)
H, W = ih.shape
igt1 = cv2.imread('HR.png', cv2.IMREAD_GRAYSCALE)   # igt shape :  (256, 256) should we perform Grayscale??? 
igt = igt1.astype(np.float64)
il = cv2.resize(igt, (H//4, W//4), interpolation = cv2.INTER_LINEAR)   # (64, 64)   ## Resize the picture => downsampling or upsampling


############################# Down_sampling ###################################### 

iteration = 10
alpha = 0.9     #0.9    => 0.009490787553010197

idt = ih  # initial condition      # (256,256)
for i in range(0,iteration):
    idt = cv2.resize(idt, (H//4, W//4), interpolation = cv2.INTER_LINEAR)   #(64,64)     , # bilinear
    grad = idt - il #(64,64)
    grad = cv2.resize(grad, (H, W), interpolation = cv2.INTER_LINEAR)
    idt = cv2.resize(idt, (H, W),interpolation = cv2.INTER_LINEAR)
    idt = idt - alpha*grad
    MSE1 = np.mean((igt-idt)**2/(H*W))
    PSNR1 = 10 * np.log10(1**2/MSE1)
    print(PSNR1)
R = 1

# upsampled.png PSNR
MSE_upsample = np.mean((igt-ih)**2/(H*W))
PSNR_upsample = 10 * np.log10(R**2/MSE_upsample)
print("upsampled : ",PSNR_upsample) ### 18.043...

##print(idt)

##min1 = np.min(subs)
##max1 = np.max(subs)
MSE = np.mean((igt-idt)**2/(H*W))
PSNR = 10 * np.log10(R**2/MSE)
##print(MSE)
print("approximated : ",PSNR)


################################    Task 2     ############################################

"""
##cv2.imread, cv2.imwrite, cv2.cvtColor, cv2.Laplacian, cv2.Sobel, cv2.resize

ih =  cv2.imread('upsampled.png', cv2.IMREAD_GRAYSCALE)   # ih shape :  (256, 256) as initial high resolution(ih0)
H, W = ih.shape
igt = cv2.imread('HR.png', cv2.IMREAD_GRAYSCALE)   # igt shape :  (256, 256)
il = cv2.resize(igt, (H//4, W//4))   # (64, 64)   ## Resize the picture => downsampling or upsampling

gamma = 6


## Normalization https://light-tree.tistory.com/132

gul_org = abs(cv2.Sobel(ih,ddepth = cv2.CV_8U, dx = 1, dy = 0)) + abs(cv2.Sobel(ih,ddepth = cv2.CV_8U, dx = 0, dy = 1))   # np.max(gul_org) = 255, np.min(gul_org) = 0 
gul = gul_org/np.max(gul_org) + 1e-10               # Why does np.max(gul_org) which is summation of x_direction and y_direction not exceed 255

lap_iu = cv2.Laplacian(ih ,ddepth = cv2.CV_8U)   #CV_8U   CV_64F    (256, 256)


gt = gul - lap_iu/np.max(lap_iu)   # np.max(lap_iu) = 118   np.min(lap_iu) = 0  np.min(gt) = -0.97457, np.max(gt) = 1.0000000001
gt = np.clip(gt,0,1)

lap_it = gamma * lap_iu * (gt/gul)
lap_it = np.clip(lap_it, 0.0, 255.0)
lap_it = cv2.resize(lap_it, (H//4, W//4)) # (64,64)

iteration = 600

alpha = 2
beta = 0.0001

idt = ih  # initial condition      # (256,256)
for i in range(0, iteration):
    lap_iht = cv2.Laplacian(ih, ddepth = cv2.CV_8U)   # (256, 256)
    idt = cv2.resize(idt, (H//4, W//4))
    
    lap_iht = cv2.resize(lap_iht, (H//4, W//4))
    
    grad = idt - il - beta * (lap_iht - lap_it)
    
    idt = idt - alpha * grad
    idt = cv2.resize(idt, (H, W))

print(idt)
R = 1
MSE = np.mean((igt - idt)**2/(H*W))
PSNR = 10 * np.log10(R**2/MSE)
print(MSE)
print(PSNR)


"""






