import sys
import cv2 as cv
import numpy as np
import ctypes
import keyboard


keys = [['A', 'Up', 'B'],
        ['Left', 'Down', 'Right']]

font = cv.FONT_HERSHEY_SIMPLEX

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

        light_yellow = np.array([23, 100, 100])
        dark_yellow = np.array([40, 255, 255])
        light_green = np.array([70, 100, 20])
        dark_green = np.array([90, 255, 255])

        light_color = light_green
        dark_color = dark_green

        mask = cv.inRange(hsv_img, light_color, dark_color)
        points = cv.findNonZero(mask)

        res = cv.bitwise_and(img, img, mask=mask)

        dimensions = res.shape
        width = dimensions[1]
        height = dimensions[0]
        xsec = int(width/3)
        ysec = int(height/2)

        if points is not None:
            avg = np.mean(points, axis=0)[0]
            cv.circle(res, (int(avg[0]), int(avg[1])), 3, (255,0,0), 3)

            keyboard.write(keys[int(avg[1]/ysec)][int(avg[0]/xsec)])
            keyboard.write('\n')

        for i in range(2):
            for j in range(3):
                cv.putText(res, keys[i][j], (xsec*j + 10, ysec*i + 40), font, 0.7, (255,255,255), 2, cv.LINE_AA)

        cv.line(res, (xsec, 0), (xsec, height), (255,255,255), 5)
        cv.line(res, (2*xsec, 0), (2*xsec, height), (255,255,255), 5)
        cv.line(res, (0, ysec), (width, ysec), (255,255,255), 5)

        cv.imshow("Raw", img)
        cv.imshow("Yellow", res)


        #cv2.imshow('my webcam', img)
        if cv.waitKey(1) == 27: 
            break  # esc to quit
    #cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True, mode=1)


if __name__ == '__main__':
    try:
        main()
    except(KeyboardInterrupt):
        print('bai')
