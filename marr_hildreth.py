import cv2
img=cv2.imread("image.webp")
img_blur=cv2.GaussianBlur(img,(3,3),0)
img_gray=cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
img_filtered=cv2.Laplacian(img_gray,cv2.CV_64F,ksize=3) #using built-in function
cv2.imshow("Original",img)
cv2.waitKey(0)
cv2.imshow("Gaussian Blurred",img_blur)
cv2.waitKey(0)
cv2.imshow("Laplacian",img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

