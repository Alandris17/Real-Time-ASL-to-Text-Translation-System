import cv2
import time
import os

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
folder = "Data/Backspace"
counter = 0
while True:
    img = cap.read()[1]
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        if not os.path.exists(folder):
            os.makedirs(folder)
        image_name = folder + '/' + 'Image_' + str(time.time())
        cv2.imwrite(image_name + '.jpg', img)
        print(counter)
    if key == ord("q"):
        break

