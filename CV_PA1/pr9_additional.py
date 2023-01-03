import cv2       ##cv2.imread, cv2.imwrite, cv2.cvtColor, cv2.Laplacian, cv2.Sobel, cv2.resize
import numpy as np
##################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##from cv2.ximgproc import guidedFilter
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

##best_parameter : alpha = 0.9

iteration = 100
alpha = 1.9     #0.9    => 0.009490787553010197                best_parameter => alpha = 0.9(20.23) ,2(20.55), check 1.9 ~ 2.0    , when alpha 0.15, chek the plot 
# alpha under 1 is localized optimization
idt = ih  # initial condition      # (256,256)
plt.figure(1)
plt.title("PSNR")
PSNR_list_task1 = []
    
for i in range(0,iteration):
    idt = cv2.resize(idt, (H//4, W//4), interpolation = cv2.INTER_LINEAR)   #(64,64)     , # bilinear
    grad = idt - il #(64,64)
    grad = cv2.resize(grad, (H, W), interpolation = cv2.INTER_LINEAR)
    idt = cv2.resize(idt, (H, W),interpolation = cv2.INTER_LINEAR)
    idt = idt - alpha*grad
    MSE1 = np.mean((igt-idt)**2/(H*W))
    PSNR1 = 10 * np.log10(1**2/MSE1)
    PSNR_list_task1.append(PSNR1)
    plt.scatter(x = i,y = PSNR1)
plt.plot(range(0,iteration), PSNR_list_task1)
    
R = 1

plt.show(block = False)
# upsampled.png PSNR
MSE_upsample = np.mean((igt-ih)**2/(H*W))
PSNR_upsample = 10 * np.log10(R**2/MSE_upsample)
print("upsampled : ",PSNR_upsample) ### 18.043...

##print(idt)

##min1 = np.min(subs)
##max1 = np.max(subs)
MSE_task1 = np.mean((igt-idt)**2/(H*W))
PSNR_task1 = 10 * np.log10(R**2/MSE_task1)
##print(MSE)
print("task1_PSNR : ",PSNR_task1)

##################################################

##guided = cv2.GuidedFilter(idt)    
##cv2.imshow("image",img)    
##cv2.imshow("guided filtering",guided)    
##cv2.waitKey()

##################################################



################################    Task 2     ############################################


##cv2.imread, cv2.imwrite, cv2.cvtColor, cv2.Laplacian, cv2.Sobel, cv2.resize

ih =  cv2.imread('upsampled.png', cv2.IMREAD_GRAYSCALE)   # ih shape :  (256, 256) as initial high resolution(ih0)
H, W = ih.shape
igt = cv2.imread('HR.png', cv2.IMREAD_GRAYSCALE)   # igt shape :  (256, 256)
il = cv2.resize(igt, (H//4, W//4))   # (64, 64)   ## Resize the picture => downsampling or upsampling

gamma = 6


## Normalization https://light-tree.tistory.com/132

gh0_org = abs(cv2.Sobel(ih,ddepth = -1, dx = 1, dy = 0)) + abs(cv2.Sobel(ih,ddepth = -1, dx = 0, dy = 1))   # np.max(gul_org) = 255, np.min(gul_org) = 0 
gh0 = gh0_org/np.max(gh0_org) + 1e-10               # Why does np.max(gul_org) which is summation of x_direction and y_direction not exceed 255

lap_ih0 = cv2.Laplacian(ih ,ddepth =-1)   #CV_8U   CV_64F    (256, 256)


gt = gh0 - lap_ih0/np.max(lap_ih0)   # np.max(lap_iu) = 118   np.min(lap_iu) = 0  np.min(gt) = -0.97457, np.max(gt) = 1.0000000001
gt = np.clip(gt,0,1)      # resonable? 

lap_it = gamma * lap_ih0 * (gt/gh0)
lap_it = np.clip(lap_it, 0.0, 255.0)

##best_parameter : alpha = 0.01, beta = 0.001, iteration = 1000

iteration = 100

alpha = 2
beta = 0.001

iht = ih  # initial condition      # (256,256)
plt.figure(2)
plt.title("PSNR")
PSNR_list_task2 = []
for i in range(0, iteration):
    lap_iht = cv2.Laplacian(iht, ddepth = -1)   # (256, 256)
    lap_iht = np.clip(lap_iht, 0.0, 255.0)
    
    idt = cv2.resize(iht, (H//4, W//4))        # (64, 64)
    grad = idt - il #(64,64)
    grad = cv2.resize(grad, (H, W), interpolation = cv2.INTER_LINEAR)
    grad = grad - beta * (lap_iht - lap_it)
    
    iht = iht - alpha * grad
    iht = np.clip(iht, 0, 255)
    
    MSE2 = np.mean((igt-iht)**2/(H*W))
    PSNR2 = 10 * np.log10(1**2/MSE2)
    PSNR_list_task2.append(PSNR2)
    plt.scatter(x = i,y = PSNR2)

plt.plot(range(0,iteration), PSNR_list_task2)

plt.show(block = False)

##print(iht)
R = 1
MSE_task2 = np.mean((igt - iht)**2/(H*W))
PSNR_task2 = 10 * np.log10(R**2/MSE_task2)
##print(MSE)
print("task2_PSNR : ", PSNR_task2)







