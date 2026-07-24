import cv2
image=cv2.imread("modern-building-with-closed-windows.webp")
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create()
kp=sift.detect(gray_image,None)
img=cv2.drawKeypoints(gray_image,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("sift detector image with keypoints",img)
cv2.waitKey(0)
cv2.destroyAllWindows()