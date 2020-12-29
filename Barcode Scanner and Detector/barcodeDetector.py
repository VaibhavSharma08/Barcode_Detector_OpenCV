import numpy as np  # importing numpy library
import cv2  # importing OpenCV library
import os
from sys import exit

""" I have followed an approach which involves:
        1. Cropping the image a bit in order to reduce the surrounding areas
        2. Converting the image to grayscale
        2. Enhancing its edges
        3. Blurring it using bilateralFilter (instead of GaussianBlur) so that edges are not lost
        4. Applying a threshold using simple thresholding
        5. Transforming the image by closing it, eroding it and then highly dilating it
        6. Searching for the boundaries and contours and drawing the box around it
    
    This approach is different from using Canny Edge filtering and then using Hough transform.
    The reason why I did not go for this approach was because this approach wasn't able to initially detect the lines 
    for me.
    So, I went for the above approach and implemented it.
    Cropping the image is an important part since it reduces the surrounding areas which might reduce the accuracy of 
    the detection.
"""

x_startCoordinate = 250  # Starting x-coordinate for crop
y_startCoordinate = 150  # Starting y-coordinate for crop
heightOfCrop = 500  # height of the crop
widthOfCrop = 600  # width of the crop


def imageReader(name):  # Function for reading the image
    image = cv2.imread(name)
    return image


def convertGrayscale(croppedImage):  # Function for converting the image to grayscale
    gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
    return gray


def edge_and_blur(gray):  # Function for enhancing the edges and blurring the image
    enhancedEdges = cv2.Laplacian(gray, ddepth=cv2.CV_8U, ksize=3, scale=1,
                                  delta=0)  # Edges being enhanced using Laplacian operator
    blurred = cv2.bilateralFilter(enhancedEdges, 10, 100, 100)  # Image being blurred through a Bilateral Filter
    return blurred


def simpleThresholding(blurred):  # Function for applying Simple Threshold
    (_, threshholdedImage) = cv2.threshold(blurred, 75, 205, cv2.THRESH_BINARY)
    return threshholdedImage


def morphologicalTransform(thresh):  # Function for eroding and dilating the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    transformedImage = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    transformedImage = cv2.erode(transformedImage, None, iterations=4)
    transformedImage = cv2.dilate(transformedImage, None, iterations=10)
    return transformedImage


def search_for_contours_and_draw(transformedImage):  # Function for drawing outlines around the detected barcode
    global cropped, original
    (contourPoints, _) = cv2.findContours(transformedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contourPoints == []:  # In case the barcode cannot be detected, exit with a message
        return None
    c = sorted(contourPoints, key=cv2.contourArea, reverse=True)[0]
    rectangleBorder = cv2.minAreaRect(c)  # Connecting the points
    markerBox = np.int0(cv2.boxPoints(rectangleBorder))
    cv2.drawContours(cropped, [markerBox], -1, (0, 255, 0), 3)
    printCoordinates(markerBox)
    return original


def displayPicture(originalImage):  # Function for displaying the image
    if originalImage is None:
        print("Couldn't find")
        return
    imageToBeDisplayed = cv2.resize(originalImage, (960, 540))
    cv2.imshow("Detected Barcode", imageToBeDisplayed)
    cv2.waitKey(0)
    writeToFile = cv2.imwrite("detected.jpg", originalImage)


def printCoordinates(markerBox):  # Function for printing the coordinates
    global x_startCoordinate, y_startCoordinate, widthOfCrop, heightOfCrop
    initialCoordinates = np.array([[x_startCoordinate, y_startCoordinate], [x_startCoordinate, y_startCoordinate],
                                   [x_startCoordinate, y_startCoordinate], [x_startCoordinate, y_startCoordinate]])
    markerBox += initialCoordinates  # Adding the initial coordinates to the box corners in order to bring the
    print(markerBox)  # coordinates back into the coordinate system of the original image


"""
# Code for using all the above functions and detecting barcode
original = imageReader("barcode9.jpg")  # I have used image as this name. Add own name for testing another image
displayOriginal = original.copy()
cropped = original[y_startCoordinate:y_startCoordinate + heightOfCrop, x_startCoordinate:x_startCoordinate + widthOfCrop]
grayscale = convertGrayscale(cropped)
blurred = edge_and_blur(grayscale)
binary = simpleThresholding(blurred)
transformed = morphologicalTransform(binary)
detected = search_for_contours_and_draw(transformed)
displayPicture(displayOriginal)
displayPicture(detected)
"""
path = "D:\Computer Vision + Machine Learning\OpenCV\Dataset2\\"
directory = os.fsencode(path)

for files in os.listdir(directory):
    print(files)
    filename1 = os.fsdecode(files)
    filename = path + filename1
    print(filename)
    original = imageReader(filename)
    # I have used image as this name. Add own name for testing another image
    displayOriginal = original.copy()
    cropped = original[y_startCoordinate:y_startCoordinate + heightOfCrop,
              x_startCoordinate:x_startCoordinate + widthOfCrop]
    grayscale = convertGrayscale(cropped)
    blurred = edge_and_blur(grayscale)
    binary = simpleThresholding(blurred)
    transformed = morphologicalTransform(binary)
    detected = search_for_contours_and_draw(transformed)
    displayPicture(displayOriginal)
    displayPicture(detected)

cv2.destroyAllWindows()
