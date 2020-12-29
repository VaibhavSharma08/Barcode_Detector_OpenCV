from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2

im = cv2.imread("barcode5.jpg")
decodedObjects = pyzbar.decode(im)
for obj in decodedObjects:
    print('Type : ', obj.type)
    print('Data : ', obj.data, '\n')

cv2.imshow("Detected Barcode", im)
