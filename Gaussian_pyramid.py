import cv2
import numpy as np
image=cv2.imread('modern-building-with-closed-windows.webp')
print(len(image))
##downsampled_image=cv2.pyrDown(image) #FUNCTION FOR DOWNSAMPLING
##upsampled_image=cv2.pyrUp(image)
##cv2.imshow("image",image)
##cv2.waitKey(0)
##cv2.imshow("downsampled_image",downsampled_image)
##cv2.waitKey(0)
##cv2.imshow("upsampled image",upsampled_image )
pyramid = [image]
for i in range(3):
    image = cv2.pyrDown(image)
    print(len(image))
    pyramid.append(image)
for i in range(len(pyramid)-1,-1,-1):
    print(f"Pyramid Level{i}")
    cv2.imshow("image",pyramid[i])
    cv2.waitKey(0)

cv2.destroyAllWindows()
