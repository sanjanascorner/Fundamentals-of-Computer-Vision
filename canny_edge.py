import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("modern-building-with-closed-windows.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.astype(np.float32)

gaussian = np.array([
    [1, 4, 6, 4, 1],
    [4,16,24,16,4],
    [6,24,36,24,6],
    [4,16,24,16,4],
    [1, 4, 6, 4, 1]
], dtype=np.float32)

gaussian = gaussian / 256

blur = cv2.filter2D(img, -1, gaussian)

sobel_x = np.array([[-1,0,1],
                   [-2,0,2],
                    [-1,0,1]], dtype=np.float32)

sobel_y = np.array([[-1,-2,-1],
    [0,0,0],
    [1,2,1]
], dtype=np.float32)

Gx = cv2.filter2D(blur, -1, sobel_x)
Gy = cv2.filter2D(blur, -1, sobel_y)

mag= np.sqrt(Gx**2 + Gy**2)
mag= mag/ mag.max() * 255

angle = np.degrees(np.arctan2(Gy, Gx))
angle[angle < 0] += 180

rows, cols = mag.shape
nms = np.zeros((rows, cols), dtype=np.float32)

for i in range(1, rows - 1):
    for j in range(1, cols - 1):

        direction = angle[i, j]

        if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
            before = mag[i, j-1]
            after = mag[i, j+1]

        elif 22.5 <= direction < 67.5:
            before = mag[i-1, j+1]
            after = mag[i+1, j+1]

        elif 67.5 <= direction < 112.5:
            before = mag[i - 1, j]
            after = mag[i + 1, j]

        else:
            before = mag[i - 1, j - 1]
            after = mag[i + 1, j + 1]

        if mag[i, j] >= before and mag[i, j] >= after:
            nms[i, j] = mag[i, j]
        else:
            nms[i, j] = 0

low = 50
high = 100

strong = 255
weak = 75

edges = np.zeros((rows, cols), dtype=np.uint8)

edges[nms >= high] = strong
edges[(nms >= low) & (nms < high)] = weak

for i in range(1, rows - 1):
    for j in range(1, cols - 1):

        if edges[i, j] == weak:

            if (edges[i-1, j-1] == strong or edges[i-1, j] == strong or edges[i-1, j+1] == strong or 
                edges[i, j-1] == strong or edges[i, j+1] == strong or edges[i+1, j-1] == strong or
                edges[i+1, j] == strong or edges[i+1, j+1] == strong ):
                edges[i, j] = strong
            else:
                edges[i, j] = 0

cv2.imshow("original", img.astype(np.uint8))
cv2.imshow("Gaussian Blur", blur.astype(np.uint8))
cv2.imshow("Gradient Magnitude", mag.astype(np.uint8))
cv2.imshow("Non-Maximum Suppression", nms.astype(np.uint8))
cv2.imshow("Final Canny Edge", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()