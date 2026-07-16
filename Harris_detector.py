import cv2
import matplotlib.pyplot as plt
img=cv2.imread("modern-building-with-closed-windows.webp")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_Harris=cv2.cornerHarris(img_gray,11,13,0.01)
img_dest=cv2.dilate(img_Harris,None)
img[img_dest>0.01 * img_dest.max()]=[255,0,0] #BGR(Blue,Green,Red) Mark the point as red
plt.imshow(img)
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()