
import glob
import os
import random

# import torchvision
import cv2
import shutil

def create_train_val_split_folder(path):
    all_categories = os.listdir(path)
    print("all categories >>", all_categories)
    os.makedirs("./dataset",exist_ok=True)
    os.makedirs("./dataset/train/",exist_ok=True)
    os.makedirs("./dataset/valid/", exist_ok=True)
    os.makedirs("./dataset/test/", exist_ok=True)

    for category in sorted(all_categories):
        os.makedirs(f"./dataset/train/{category}",exist_ok=True)
        all_image = os.listdir(f"./sobun/{category}/")
        #print("all_image>>", all_image)
        for image in random.sample(all_image, int(0.8*len(all_image))):
            #shutil의 인자값 = shutil.move(기존경로, 옮길 경로)
            shutil.move(f"./sobun/{category}/{image}",f"./dataset/train/{category}/{category}2_{image}")
    # for category in sorted(all_categories):
    #     os.makedirs(f"./dataset/val/{category}", exist_ok=True)
    #     all_image = os.listdir(f"./game/{category}/")
    #     for image in random.sample(all_image, int(0.5*len(all_image))):
    #         shutil.move(f"./game/{category}/{image}",f"./dataset/val/{category}/")
    for category in sorted(all_categories):
        os.makedirs(f"./dataset/valid/{category}", exist_ok=True)
        all_image = os.listdir(f"./sobun/{category}/")
        for image in all_image:
            shutil.move(f"./sobun/{category}/{image}",f"./dataset/valid/{category}/{category}2_{image}")

if __name__ =="__main__":
    path = "./sobun"
    # image_size(path)
    create_train_val_split_folder(path)