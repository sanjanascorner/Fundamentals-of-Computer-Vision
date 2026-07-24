import cv2
img=cv2.imread("modern-building-with-closed-windows.webp")
#cv2.imshow('Original',img)
#cv2.waitKey(0)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_blur=cv2.GaussianBlur(img_gray,(3,3),0)

#canny edge 
edges1=cv2.Canny(image=img_blur,threshold1=20,threshold2=30)
edges2=cv2.Canny(image=img_blur,threshold1=100,threshold2=200)
cv2.imshow('Canny edge detection with threshold of 50 & 100',edges1)
cv2.imshow('Canny edge detection with threshold of 100 & 200',edges2)
cv2.waitKey(0)

cv2.destroyAllWindows()