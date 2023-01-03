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


R = 1


#############################       Task_1      ############################################

############################# Load_data ##########################################
ih1 =  cv2.imread('upsampled.png', cv2.IMREAD_GRAYSCALE)   # ih shape :  (256, 256) as initial high resolution(ih0)
ih = ih1.astype(np.float64)
H, W = ih.shape
igt1 = cv2.imread('HR.png', cv2.IMREAD_GRAYSCALE)   # igt shape :  (256, 256) should we perform Grayscale??? 
igt = igt1.astype(np.float64)
il = cv2.resize(igt, (H//4, W//4), interpolation = cv2.INTER_LINEAR)   # (64, 64)   ## Resize the picture => downsampling or upsampling

###########################  upsampled.png PSNR  ###################################

R = 1

MSE_upsample = np.mean((igt-ih)**2/(H*W))
PSNR_upsample = 10 * np.log10(R**2/MSE_upsample)
print("\n")
print("######################## upsampled image ########################")
print("\n")
print("upsampled_MSE : {:.5f}".format(MSE_upsample))
print("upsampled_PSNR : {:.3f}".format(PSNR_upsample)) ### 18.043...
print("\n")
######################################################################################

##best_parameter : alpha = 0.9
##alpha = 2     #0.9    => 0.009490787553010197                best_parameter => alpha = 0.9(20.23) ,2(20.55), check 1.9 ~ 2.0    , when alpha 0.15, chek the plot 
# alpha under 1 is localized optimization
##idt = ih  # initial condition      # (256,256)

print("######################## Task 1 ########################")
print("\n")

iteration = 300              # 100
print("The number of iteration : ",iteration)
plt.figure(1, figsize = (8,6))
plt.title("Task1 PSNR according to learning_rate")
plt.xlabel('iteration')
plt.ylabel('PSNR')

PSNR_list_task1 = []
MSE_list_tast1 = []
PSNR_task1_result = []
alpha_list = np.arange(1.4,2.2,0.1)
##alpha_list = [0.1, 0.2, 0.3,1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
j = 0
    
for alpha in alpha_list:
    idt = ih
    for i in range(0,iteration):
        idt = cv2.resize(idt, (H//4, W//4), interpolation = cv2.INTER_LINEAR)   #(64,64)     , # bilinear
        grad = idt - il #(64,64)
        grad = cv2.resize(grad, (H, W), interpolation = cv2.INTER_LINEAR)
        idt = cv2.resize(idt, (H, W),interpolation = cv2.INTER_LINEAR)
        idt = idt - alpha*grad
        MSE1 = np.mean((igt-idt)**2/(H*W))
        PSNR1 = 10 * np.log10(1**2/MSE1)
        PSNR_list_task1.append(PSNR1)
        MSE_list_tast1.append(MSE1)
    plt.plot(range(0,iteration), PSNR_list_task1, label = 'lr = {:.1f}'.format(alpha_list[j]))
    PSNR_task1_result.append(PSNR_list_task1[iteration-1])
    plt.legend(loc="lower right")


    MSE_task1 = np.mean((igt-idt)**2/(H*W))
    PSNR_task1 = 10 * np.log10(R**2/MSE_task1)
    print("task1_PSNR_{:.1f}(lr) : {:.3f}".format(alpha_list[j],PSNR_task1))
    PSNR_list_task1.clear()
    MSE_list_tast1.clear()
    j= j + 1
   
plt.ylim([17, 21])
plt.show(block = False)


max_PSNR_task1 = max(PSNR_task1_result)
index = PSNR_task1_result.index(max_PSNR_task1)

print("\n")
print("######## Find the optimal number of iteration ##########")
print("\n")
print("task1_PSNR_{:.1f}(lr) is the best when the same iteration : {:.3f}".format(alpha_list[index], max_PSNR_task1))

########################## Find the optimal iteration and PSNR ######################


alpha = alpha_list[index]

j = 0

iteration = 300         # 200 is good

plt.figure(2, figsize = (8,6))
plt.title("Task1 PSNR using {:1f} learning_rate".format(alpha))
plt.xlabel('iteration')
plt.ylabel('PSNR')

idt = ih
for i in range(0,iteration):
    idt = cv2.resize(idt, (H//4, W//4), interpolation = cv2.INTER_LINEAR)   #(64,64)     , # bilinear
    grad = idt - il #(64,64)
    grad = cv2.resize(grad, (H, W), interpolation = cv2.INTER_LINEAR)
    idt = cv2.resize(idt, (H, W),interpolation = cv2.INTER_LINEAR)
    idt = idt - alpha*grad
    MSE1 = np.mean((igt-idt)**2/(H*W))
    PSNR1 = 10 * np.log10(1**2/MSE1)
    PSNR_list_task1.append(PSNR1)
    MSE_list_tast1.append(MSE1)

plt.plot(range(0,iteration), PSNR_list_task1, label = 'lr = {:.1f}'.format(alpha))

task1_max_PSNR = max(PSNR_list_task1)
index = PSNR_list_task1.index(task1_max_PSNR)

plt.legend(loc="lower right")

print("\n")
print("Max PSNR is {:.3f} when iterated {} using {:.1f} learning_rate".format(task1_max_PSNR, index+1, alpha))
print("MSE : {:.5f}".format(MSE_list_tast1[index+1]))
print("\n")
print("#######################################################")
print("\n")

plt.show(block=False)




################################    Task 2     ############################################

print("######################## Task 2 ########################")
print("\n")

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

iteration = 300        #150
print("The number of iteration : ", iteration)

beta = 0.001

iht = ih  # initial condition      # (256,256)
plt.figure(3, figsize = (8,6))
plt.title("Task2 PSNR according to learning_rate")
plt.xlabel('iteration')
plt.ylabel('PSNR')
PSNR_list_task2 = []
MSE_list_tast2 = []
PSNR_task2_result = []

alpha_list = np.arange(0.02,0.03,0.001)


k = 0

for alpha in alpha_list:
    iht = ih  # initial condition      # (256,256)

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
        MSE_list_tast2.append(MSE2)

    plt.plot(range(0,iteration), PSNR_list_task2, label = 'lr = {:.3f}'.format(alpha_list[k]))
    PSNR_task2_result.append(PSNR_list_task2[iteration-1])
    plt.legend(loc="lower right")

    MSE_task2 = np.mean((igt - iht)**2/(H*W))
    PSNR_task2 = 10 * np.log10(R**2/MSE_task2)
    print("task2_PSNR_{:.3f}(lr) : {:.3f}".format(alpha_list[k],PSNR_task2))
    PSNR_list_task2.clear()
    MSE_list_tast2.clear()
    k= k+1


plt.show(block = False)

task2_max_PSNR = max(PSNR_task2_result)
index = PSNR_task2_result.index(task2_max_PSNR)

print("\n")
print("######## Find the optimal number of iteration ##########")
print("\n")
print("task2_PSNR_{:.3f}(lr) is the best when the same iteration : {:.3f}".format(alpha_list[index], task2_max_PSNR))

########################## Find the optimal iteration and PSNR ######################

alpha = alpha_list[index]

k = 0

iht = ih  # initial condition      # (256,256)

iteration = 300         # 200 is good

plt.figure(4, figsize = (8,6))
plt.title("Task2 PSNR using {} learning_rate".format(alpha))
plt.xlabel('iteration')
plt.ylabel('PSNR')

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
    PSNR_list_task2.append(PSNR2)               ### result according to iteration
    MSE_list_tast2.append(MSE2)

plt.plot(range(0,iteration), PSNR_list_task2, label = 'lr = {:.3f}'.format(alpha_list[k]))

task2_max_PSNR = max(PSNR_list_task2)
index = PSNR_list_task2.index(task2_max_PSNR)

print("\n")
print("Max PSNR is {:.3f} when iterated {} using {:.3f} learning_rate".format(task2_max_PSNR, index+1, alpha))
print("MSE : {:.5f}".format(MSE_list_tast2[index+1]))
print("\n")
print("#######################################################")
print("\n")

plt.legend(loc="lower right")


plt.show(block=False)


