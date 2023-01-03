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
ih =  cv2.imread('upsampled.png', cv2.IMREAD_GRAYSCALE)   # ih shape :  (256, 256) as initial high resolution(ih0)
H, W = ih.shape
igt = cv2.imread('HR.png', cv2.IMREAD_GRAYSCALE)   # igt shape :  (256, 256) should we perform Grayscale??? 
il = cv2.resize(igt, (H//4, W//4))   # (64, 64)   ## Resize the picture => downsampling or upsampling


############################# Down_sampling ###################################### 

iteration = 1000
alpha = 0.01

idt = ih  # initial condition      # (256,256)
for i in range(0,iteration):
    idt = cv2.resize(idt, (H//4, W//4))   #(64,64)
    grad = idt - il
    idt = idt - alpha*grad
    idt = cv2.resize(idt, (H, W))

print(idt)
R = 1
MSE = np.mean((igt - idt)**2/(H*W))
PSNR = 10 * np.log10(R**2/MSE)
print(MSE)
print(PSNR)


################################    Task 2     ############################################

"""
##cv2.imread, cv2.imwrite, cv2.cvtColor, cv2.Laplacian, cv2.Sobel, cv2.resize

ih =  cv2.imread('upsampled.png', cv2.IMREAD_GRAYSCALE)   # ih shape :  (256, 256) as initial high resolution(ih0)
H, W = ih.shape
igt = cv2.imread('HR.png', cv2.IMREAD_GRAYSCALE)   # igt shape :  (256, 256)
il = cv2.resize(igt, (H//4, W//4))   # (64, 64)   ## Resize the picture => downsampling or upsampling

gamma = 6

##a = cv2.Sobel(ih,ddepth = cv2.CV_8U, dx = 1, dy = 0)
##b = cv2.Sobel(ih,ddepth = cv2.CV_8U, dx = 0, dy = 1)
##c = a+b
##d = abs(c)   # np.array_equal(c,d) = True
##
##plt.figure(1)
##plt.imshow(a)
##plt.show(block=False)
##plt.figure(2)
##plt.imshow(b)
##plt.show(block=False)
##plt.figure(3)
##plt.imshow(c)
##plt.show(block=False)
##plt.figure(4)
##plt.imshow(d)
##plt.show(block=False)

## Normalization https://light-tree.tistory.com/132

gul_org = abs(cv2.Sobel(ih,ddepth = cv2.CV_8U, dx = 1, dy = 0)) + abs(cv2.Sobel(ih,ddepth = cv2.CV_8U, dx = 0, dy = 1))   # np.max(gul_org) = 255, np.min(gul_org) = 0 
gul = gul_org/np.max(gul_org) + 1e-10               # Why does np.max(gul_org) which is summation of x_direction and y_direction not exceed 255

lap_iu = cv2.Laplacian(ih ,ddepth = cv2.CV_8U)   #CV_8U   CV_64F

a = cv2.Laplacian(ih ,ddepth = cv2.CV_8U)

##b = cv2.Laplacian(ih ,ddepth = cv2.CV_64F)

##plt.figure(1)
##plt.imshow(a)
##
##plt.figure(2)
##plt.imshow(b)
##
##plt.show()


gt = gul - lap_iu/np.max(lap_iu)   # np.max(lap_iu) = 118   np.min(lap_iu) = 0  np.min(gt) = -0.97457, np.max(gt) = 1.0000000001

gt = np.clip(gt,0,1)

lap_it = gamma * lap_iu * (gt/gul)


lap_it = np.clip(lap_it, 0.0, 255.0)


iteration = 5

alpha = 2
beta = 0.0001

idt = ih  # initial condition      # (256,256)
for i in range(0, iteration):
    lap_iht = cv2.Laplacian(ih, ddepth = cv2.CV_8U)   # (256, 256)
    idt = cv2.resize(idt, (H//4, W//4))
    grad = idt - il - beta * (lap_iht - lap_it)
    idt = idt - alpha * grad


    

##    idt = cv2.resize(idt, (H//4, W//4))   #(64,64)
##    grad = idt - il
##    idt = idt - alpha*grad
##    idt = cv2.resize(idt, (H, W))



"""







