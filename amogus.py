import pyautogui

while(True):
    location = pyautogui.locateOnScreen('impostor.png', confidence = 0.5)
    if(location != None):
        print("Impostor!")
        break
    location = pyautogui.locateOnScreen('crewmate.png', confidence = 0.5)
    if(location != None):
        print("Crewmate!")
        break
    location = pyautogui.locateOnScreen('close.png', grayscale=True, confidence = 0.7)
    if(location != None):
        print("X!")
        break