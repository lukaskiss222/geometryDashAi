import time
import cv2
import mss
import numpy as np
import time
import pyautogui
import grabimage
import os
import getkeys

fps = 60 
file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exits, loading previous data')
    training_data = list(np.load(file_name))
else:
    print('File does not exits, starting fresh')
    training_data = []

def keys_to_output(keys):
    output = [0]
    if "Up" in keys:
        output[0] = 1

    return output


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

def jump():
    pyautogui.press("Up")


def main():
    global training_data
    local_data = []

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    shots = grabimage.ScreenShot()
    keys = getkeys.GetKeys(new_keys = ["Up", 'a', 'r', 's'])
    for shot in shots.images():
        new_screen = cv2.cvtColor(shot, cv2.COLOR_RGB2GRAY)
        new_screen = cv2.resize(new_screen, (80,60))
        cv2.imshow('window', new_screen)
        ke = keys.key_check()
        output = keys_to_output(ke)
        local_data.append([new_screen, output])
        
         
        if (len(local_data) + len(training_data)) % 500 == 0:
            print(len(training_data) + len(local_data))
        
        if 'a' in ke:
            print("Appending Data: ", len(local_data))
            training_data +=local_data
            local_data = []
            time.sleep(0.01)
        
        if 'r' in ke:
            print("Reseting data")
            local_data = []
            time.sleep(0.01)
        
        if 's' in ke:
            print("Saving Data: ",len(training_data) )
            np.save(file_name, training_data)
            time.sleep(0.01)

        # Press q to quit
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break

if __name__ == "__main__":
    main()
