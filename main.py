from os import listdir, getcwd
from re import match
import sys
from threading import current_thread
import cv2 as cv
import numpy as np
import ctypes
import keyboard
import mouse
from statistics import mean


MACROS_FILE = 'macros.txt'
AMIIBO_DIR = 'amiibo'


keys = [['enter', 'w', 'esc'],
        ['a', 's', 'd']]

try:
    with open(MACROS_FILE) as file:
        macros = [line.rstrip() for line in file]

    orb = cv.ORB_create()

    amiibo_names = [figure for figure in listdir(getcwd()) if figure[0] == 'a']
    amiibo = []
    for index in range(0, len(amiibo_names)):
        figure = {}
        figure['name'] = amiibo_names[index]
        figure['img'] = cv.imread(figure['name'], 0)
        figure['kp'], figure['des'] = orb.detectAndCompute(figure['img'], None)
        figure['macro'] = macros[index]
        amiibo.append(figure)
except IndexError:
    print('# of macros != # of images')
    sys.exit(0)

font = cv.FONT_HERSHEY_SIMPLEX

def show_webcam(mirror=False, mode=0):
    pressed_key = 'a'
    active_macro = False
    macro_frames = 0
    last_macro = -1

    cam = cv.VideoCapture(0, cv.CAP_DSHOW)
    while True:

        ret_val, orimg = cam.read()
        if not ret_val or orimg is None:
            print('Error opening image')
            exit()

        if mirror: 
            orimg = cv.flip(orimg, 1)

        orb = cv.ORB_create()
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        kp, des = orb.detectAndCompute(orimg, None)
        
        if des is None:
            print('no des')
            continue

        current_score = 0
        current_index = -1
        for index, figure in enumerate(amiibo):
            matches = bf.match(figure['des'], des)
            matches = sorted(matches, key=lambda x:x.distance)
            #res = cv.drawMatches(figure['img'], figure['kp'], orimg, kp, matches, None, flags=2)
            #cv.imshow(figure['name'], res)

            if len(matches) == 0:
                continue
            
            figure['score'] = matches[0].distance
            #print(figure['name'], ' : ', figure['score'])
            
            if figure['score'] < 20 and figure['score'] > current_score:
                current_score = figure['score']
                current_index = index

        if current_index > -1:
            #print('VICTORIA: ', amiibo[current_index]['name'])
            if not active_macro and macro_frames > 75:
                keyboard.press_and_release(amiibo[current_index]['macro'])
                print('MACRO: ', amiibo[current_index]['name'])
                macro_frames = 0
                active_macro = True
        else:
            macro_frames += 1
            active_macro = False
        
        img = cv.cvtColor(orimg, cv.IMREAD_COLOR)
        img = cv.cvtColor(orimg, cv.COLOR_BGR2RGB)

        blur = cv.blur(img,(5,5))
        blur0 = cv.medianBlur(blur,5)
        blur1 = cv.GaussianBlur(blur0,(5,5),0)
        img = cv.bilateralFilter(blur1,9,75,75)

        hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        light_yellow = np.array([23, 100, 100])
        dark_yellow = np.array([40, 255, 255])
        light_green = np.array([70, 100, 20])
        dark_green = np.array([90, 255, 255])

        light_color = light_green
        dark_color = dark_green

        kernel = np.ones((5,5), np.uint8)

        mask = cv.inRange(hsv_img, light_color, dark_color)
        mask = cv.dilate(mask, kernel, iterations=5)
        mask = cv.erode(mask, kernel, iterations=5)
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

            #keyboard.write(keys[int(avg[1]/ysec)][int(avg[0]/xsec)])
            #keyboard.write('\n')
            current_key = keys[int(avg[1]/ysec)][int(avg[0]/xsec)]
            if current_key != pressed_key:
                keyboard.release(pressed_key)
                keyboard.press(current_key)
                pressed_key = current_key

        for i in range(2):
            for j in range(3):
                cv.putText(res, keys[i][j], (xsec*j + 10, ysec*i + 40), font, 0.7, (255,255,255), 2, cv.LINE_AA)

        cv.line(res, (xsec, 0), (xsec, height), (255,255,255), 5)
        cv.line(res, (2*xsec, 0), (2*xsec, height), (255,255,255), 5)
        cv.line(res, (0, ysec), (width, ysec), (255,255,255), 5)

        cv.imshow("Raw", orimg)
        cv.imshow("Yellow", res)

        if cv.waitKey(50) == 27: 
            break  # esc to quit
    keyboard.release(pressed_key)


def main():
    show_webcam(mirror=False, mode=1)


if __name__ == '__main__':
    try:
        main()
    except(KeyboardInterrupt):
        print('bai')
