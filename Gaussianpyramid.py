import cv2
import numpy as np
img=cv2.imread("modern-building-with-closed-windows.webp")
img=img.astype(np.float32)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def gaussian_pyramid(image):
    kernel=np.array([[1,4,6,4,1],
                    [4,16,24,16,4],
                    [6,24,36,24,6], 
                    [4,16,24,16,4],
                    [1,4,6,4,1]],dtype=np.float32) #Gaussian kernel
    kernel= kernel/256.0
    
    padded = np.pad(image, ((2,2),(2,2)), mode='reflect')

    h,w=image.shape
    blurred=np.zeros((h,w),dtype=np.float32)
    for x in range(h):
        for y in range(w):
            region=padded[x:x+5,y:y+5]
            blurred[x,y]=np.sum(region*kernel)
    reduced=blurred[::2,::2]
    return reduced
def expand(image):
    kernel=np.array([[1,4,6,4,1],
                    [4,16,24,16,4],
                    [6,24,36,24,6], 
                    [4,16,24,16,4],
                    [1,4,6,4,1]],dtype=np.float32) #Gaussian kernel
    kernel= kernel/256.0
    kernel=kernel*4
    h,w=image.shape
    expanded=np.zeros((h*2,w*2),dtype=np.float32)
    expanded[::2,::2]=image
    padded = np.pad(expanded, ((2,2),(2,2)), mode='reflect')

    result = np.zeros_like(expanded)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            region = padded[i:i+5, j:j+5]
            result[i,j] = np.sum(region * kernel)

    return result

G0=gaussian_pyramid(img)
cv2.imshow("G0",G0.astype(np.uint8))
cv2.waitKey(0)
G1=gaussian_pyramid(G0)
cv2.imshow("G1",G1.astype(np.uint8))
cv2.waitKey(0)
G2=gaussian_pyramid(G1)
cv2.imshow("G2",G2.astype(np.uint8))
cv2.waitKey(0)
G3=gaussian_pyramid(G2)
cv2.imshow("G3",G3.astype(np.uint8))
cv2.waitKey(0)
#Laplacian Pyramid
L0=G0 - expand(G1)
cv2.imshow("L0",L0)
L1=G1 - expand(G2)
cv2.imshow("L1",L1)
L2=G2 - expand(G3)
cv2.imshow("L2",L2)
L3 = G3
cv2.imshow("L3",L3)

cv2.waitKey(0)
cv2.destroyAllWindows()