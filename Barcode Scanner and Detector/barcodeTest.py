import numpy as np
import cv2

originalImage = cv2.imread("barcode9.jpg")

imS = cv2.resize(originalImage, (960, 540))
cv2.imshow("Original", imS)
cv2.waitKey(0)

x_startCoordinate = 250
y_startCoordinate = 150
heightOfCrop = 500
widthOfCrop = 600
image = originalImage[y_startCoordinate:y_startCoordinate + heightOfCrop, x_startCoordinate:x_startCoordinate + widthOfCrop]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imS = cv2.resize(gray, (960, 540))
cv2.imshow("Cropped", imS)
cv2.waitKey(0)
"""
# equalize lighting
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
imS = cv2.resize(gray, (960, 540))
cv2.imshow("Lighting", imS)
cv2.waitKey(0)
"""
# edge enhancement
edge_enh = cv2.Laplacian(gray, ddepth=cv2.CV_8U,
                         ksize=3, scale=1, delta=0)
imS = cv2.resize(edge_enh, (960, 540))
cv2.imshow("Edges", imS)
cv2.waitKey(0)
retval = cv2.imwrite("edge_enh.jpg", edge_enh)

# bilateral blur, which keeps edges
blurred = cv2.bilateralFilter(edge_enh, 10, 100, 100)
"""
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
"""
imS = cv2.resize(blurred, (960, 540))
cv2.imshow("Blurred", imS)
cv2.waitKey(0)

# use simple thresholding. adaptive thresholding might be more robust
(_, thresh) = cv2.threshold(blurred, 75, 205, cv2.THRESH_BINARY)
imS = cv2.resize(thresh, (960, 540))
cv2.imshow("Thresholded", imS)
cv2.waitKey(0)
retval = cv2.imwrite("thresh.jpg", thresh)

# do some morphology to isolate just the barcode blob
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
imS = cv2.resize(closed, (960, 540))
cv2.imshow("After first close", imS)
cv2.waitKey(0)
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=10)
imS = cv2.resize(closed, (960, 540))
cv2.imshow("After morphology", imS)
cv2.waitKey(0)
retval = cv2.imwrite("closed.jpg", closed)

# find contours left in the image
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(cnts)
if(cnts==[]):
    exit("Couldn't find")
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
coor = np.array([[x_startCoordinate, y_startCoordinate], [x_startCoordinate, y_startCoordinate], [x_startCoordinate, y_startCoordinate], [x_startCoordinate, y_startCoordinate]])
box = box + coor
print(box)
imS = cv2.resize(originalImage, (960, 540))
cv2.imshow("found barcode", imS)
cv2.waitKey(0)
retval = cv2.imwrite("found.jpg", originalImage)

cv2.destroyAllWindows()
