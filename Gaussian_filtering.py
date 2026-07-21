import cv2
import numpy as np
image=cv2.imread("modern-building-with-closed-windows.webp")
kernel_size=5
sigma=1
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def generate_gaussian(size,sigma):
    kernel=np.zeros((size,size),dtype='float32')
    center=(size-1)/2.0
    for x in range(size):
        for y in range(size):
            diff_x=  x - center
            diff_y=  y - center
            exponent=-(diff_x**2+diff_y**2) / (2*sigma**2)
            kernel[x,y]= (1/2*np.pi*sigma**2)*np.exp(exponent)
    #normalised kernel
    kernel=kernel/np.sum(kernel)
    return kernel

gaussian_kernel=generate_gaussian(kernel_size,sigma)
height,width=image.shape
image_after=np.zeros((height,width),dtype='float32') #basically stores image after gaussian blur has been performed
print('After applying Gaussian Kernel')
for i in range(height-kernel_size+1):
    for j in range(width-kernel_size+1):
        region=image[i:i+kernel_size,j:j+kernel_size].astype('float32')
        convolution_result=np.sum(region*gaussian_kernel)
        image_after[i+kernel_size//2,j+kernel_size//2]= convolution_result
#converting back to uint8
image_after=np.clip(image_after,0,255).astype('uint8')
cv2.imshow("Original image",image)
cv2.imshow("Image after gaussian blur",image_after)
cv2.waitKey(0)
cv2.destroyAllWindows()

