import numpy as np
import cv2
import time
import os
from grabscreen import grab_screen
from grabkeys import key_check
from PIL import ImageGrab
from PIL import Image
from input_keys import PressKey, ReleaseKey

Q = 0x10
J = 0x24
K = 0x25
L = 0x26

def keys_to_output(keys):

    # [Q,J,K,L]
    output = [0,0,0,0]

    if 'A' in keys:
        output[0] = 1
    # elif 'W' in keys:
    #     output[1] = 1
    elif 'B' in keys:
        output[1] = 1
    elif 'N' in keys:
        output[2] = 1
    elif 'M' in keys:
        output[3] = 1

    return output


file_name = 'training_data.npy'

# if os.path.isfile(file_name):
# print('File exist, loading previous data!')
# training_data = list(np.load(file_name))
time.sleep(1)

# else:
#     print('File does not exist, starting fresh')
#     training_data = []

i=0
def main():

    global training_data
    global i
    while (True):

        i = i+1
        screen = np.array(ImageGrab.grab(bbox=(0, 120, 180, 480)))


        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (180, 360))


        keys = key_check()

        output = keys_to_output(keys)

        # print(type(training_data))
        # training_data = list(training_data)
        # training_data.append([screen, output])



        img_2 = Image.fromarray(screen)  # NumPy array to PIL image
        os.makedirs('./data', exist_ok=True)
        os.makedirs('./data/q', exist_ok=True)
        #os.makedirs('./w', exist_ok=True)
        os.makedirs('./data/j', exist_ok=True)
        os.makedirs('./data/k', exist_ok=True)
        os.makedirs('./data/l', exist_ok=True)
        if output == [1, 0, 0,0]:
            img_2.save(f'./data/q/q2_{i}.jpg','png')  # save PIL image
        # elif output == [0, 1, 0,0,0]:
        #     img_2.save(f'./w/w1_{i}.jpg','png')
        #     time.sleep(0.3)
        elif output == [0, 1, 0,0]:
            img_2.save(f'./data/j/j2_{i}.jpg','png')
        elif output == [0, 0, 1,0]:
            img_2.save(f'./data/k/k2_{i}.jpg','png')
        elif output==[0,0,0,1]:
            img_2.save(f'./data/l/l2_{i}.jpg','png')

        if output == [1, 0, 0,0]:
            PressKey(Q)
        elif output == [0, 1, 0,0]:
            PressKey(J)
        elif output == [0, 0, 1,0]:
            PressKey(K)
        elif output == [0, 0, 0,1]:
            PressKey(L)
        else:
            ReleaseKey(Q)
            ReleaseKey(J)
            ReleaseKey(K)
            ReleaseKey(L)


        #
        #
        # if len(training_data) % 2 == 0:
        #     print(training_data)
        #     training_data = np.array(training_data)#, dtype=object)
        #     print(len(training_data))
        #     np.save(file_name, training_data)
        #

main()