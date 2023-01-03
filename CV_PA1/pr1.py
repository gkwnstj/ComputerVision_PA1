import cv2

##cv2.imread, cv2.imwrite, cv2.cvtColor, cv2.Laplacian, cv2.Sobel, cv2.resize

##################################################################################
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity
##################################################################################

############################# Load_data ##########################################
ih =  cv2.imread('upsampled.png', cv2.IMREAD_GRAYSCALE)   # ih shape :  (256, 256) as initial high resolution(ih0)
print("ih shape : ", ih.shape)   
height, width = ih.shape

igt = cv2.imread('HR.png', cv2.IMREAD_GRAYSCALE)   # igt shape :  (256, 256)
print("igt shape : ", igt.shape)

############################# Down_sampling ###################################### 



il = cv2.resize(igt, (height//4, width//4))   # (64, 64)   ## Resize the picture => downsampling or upsampling
print("il shape(downsampling the igt(HR)) : ", il.shape)

iteration = 3

ih0 = cv2.resize(ih, (height//4, width//4)) ## "Upsampled.png"

##a = ih0 - il

alpha = 0.01

idt = ih0

iht = ih

for i in range(0,iteration):
##    idt = cv2.resize(idh, (height//4, width//4))  # ih = upsampled
##    grad = cv2.resize(idt - il,(height, width))
##    print("idt shape : ", idt.shape)
##    print("grad shape : ", grad.shape)
##
##    idt = idt - alpha * grad
    error = cv2.resize(ih,(height//4, width//4)) - il
    grad = cv2.resize(error, (height, width))
    print(grad)
    iht = iht + alpha * grad




