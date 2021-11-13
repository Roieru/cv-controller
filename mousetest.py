import mouse
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils
from ShapeDetector import ShapeDetector

def show_webcam(mirror=False, mode=0):
    cam = cv.VideoCapture(0, cv.CAP_DSHOW)
    while True:

        ret_val, img = cam.read()
        if mirror: 
            img = cv.flip(img, 1)
        
        img = cv.cvtColor(img, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img = cv.GaussianBlur(img, (5,5), 0)

        hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        lred1 = (110, 40, 50)
        lred2 = (150, 255, 170)

        dimensions = img.shape
        width = dimensions[1]
        height = dimensions[0]
        xsec = int(width/3)
        ysec = int(height/3)

        mask = cv.inRange(hsv_img, lred1, lred2)

        res = cv.bitwise_and(img, img, mask=mask)

        points = cv.findNonZero(mask)

        resized = imutils.resize(hsv_img, width=300)
        ratio = height / float(resized.shape[0])

        blurred = cv.GaussianBlur(resized, (5, 5), 0)

        mask = cv.inRange(blurred, lred1, lred2)

        kernel = np.ones((10,10), np.uint8)

        for _ in range(10):
            mask = cv.erode(mask, kernel, iterations=1)
            mask = cv.dilate(mask, kernel, iterations=1)

        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        sd = ShapeDetector()

        for c in cnts:
            if c.shape[0] < 50:
                continue

            shape = sd.detect(c)

            if mouse.is_pressed() and shape == "rectangle":
                mouse.release()
            elif not mouse.is_pressed() and shape == "circle":
                mouse.press()

        if points is not None:

            avg = np.mean(points, axis=0)[0]

            cv.circle(res, (int(avg[0]), int(avg[1])), 3, (0,255,0), 3)

        moveStep = (int)(cv.countNonZero(cv.cvtColor(res, cv.COLOR_RGB2GRAY)) / 3000)

        ymove = int(avg[1]/ysec) - 1
        xmove = int(avg[0]/xsec) - 1

        mouse.move(xmove * moveStep, ymove * moveStep, absolute=False)

        cv.line(res, (xsec, 0), (xsec, height), (255,255,255), 5)
        cv.line(res, (xsec*2, 0), (xsec*2, height), (255,255,255), 5)
        cv.line(res, (0, ysec), (width, ysec), (255,255,255), 5)
        cv.line(res, (0, ysec*2), (width, ysec*2), (255,255,255), 5)

        cv.imshow("Raw", res)

        plt.pause(0.05)

        #cv2.imshow('my webcam', img)
        if cv.waitKey(1) == 27: 
            break  # esc to quit
    #cv2.destroyAllWindows()

plt.show()

def main():
    show_webcam(mirror=True, mode=1)


if __name__ == '__main__':
    try:
        main()
    except(KeyboardInterrupt):
        print('bai')
