import cv2
import numpy as np
img=cv2.imread("image.webp")
cv2.imshow('Original',img)
cv2.waitKey(0)

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_blur= cv2.GaussianBlur(img_gray,(3,3),0)
#Sobel operator
sobelx=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=7)
cv2.imshow('Sobel in X axis',sobelx)
cv2.waitKey(0)
sobely=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=0,dy=1,ksize=7)
cv2.imshow('Sobel in Y axis',sobely)
cv2.waitKey(0)
sobelxy=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=1,dy=1,ksize=7)
cv2.imshow('Sobel in XY axis',sobelxy)
cv2.waitKey(0)

#prewitt operator
prewittx=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=np.float32)
prewitty=np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype=np.float32)
gx=cv2.filter2D(img_blur,-1,prewittx)
gy=cv2.filter2D(img_blur,-1,prewitty)
cv2.imshow("prewitt in x axis",gx)
cv2.waitKey(0)
cv2.imshow("prewitt in y axis",gy)
cv2.waitKey(0)


cv2.destroyAllWindows()
