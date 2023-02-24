from torchvision import models
import torch
import torch.nn as nn
from PIL import ImageGrab
import cv2
import torch.nn.functional as F
import albumentations as Al
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from input_keys import PressKey, ReleaseKey
import time

labels = {0: 'Q', 1: 'J', 2: 'K', 3: 'L'}

def ingame_predic():
    test_transform = Al.Compose(
        [
            # A.SmallestMaxSize(max_size=160),
            Al.Resize(width = 360, height = 540),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = models.mobilenet_v3_large(pretrained=True)    # mobilenet-v3
    # net.classifier[3] = nn.Linear(in_features = 1280, out_features=4)


    # net = models.efficientnet_b4(pretrained=True)   # efficientnet
    # net.classifier[1] = nn.Linear(in_features=1792,out_features=3)
    # net.load_state_dict(torch.load('./models/EFFI_output3.pt', map_location=device))
    #
    #
    # net = models.resnet18(pretrained=True)
    # #net.conv1 = nn.Linear(net.conv1.in_features, 64*3*7*7)
    # net.fc = nn.Linear(in_features=512,out_features=4)#450개로 분류하잖음
    # net.load_state_dict(torch.load('./models/resnet18.pt', map_location=device))


    # net = models.resnet50(pretrained=True)
    # #net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=7, padding=3, bias=False)
    # net.fc = nn.Linear(in_features=2048,out_features=4)#450개로 분류하잖음
    # net.load_state_dict(torch.load('./models/RESNET50_for_tetris.pt', map_location=device))

    # net = models.swin_t(weights='IMAGENET1K_V1')
    # net.head = nn.Linear(in_features=768, out_features=4)
    # net.load_state_dict(torch.load('./models/SWIN_T_TET.pt', map_location=device))

    # net = models.resnet18(pretrained=True)
    # net.fc = nn.Linear(in_features=512,out_features=4)#450개로 분류하잖음
    # net.load_state_dict(torch.load('./models/EFFI_TET.pt', map_location=devikce))

    net = models.efficientnet_b4(pretrained=True)
    net.classifier[1] = nn.Linear(in_features=1792,out_features=4)#450개로 분류하잖음
    net.load_state_dict(torch.load('./models/EFFI_TET.pt', map_location=device))

    net.to(device)
    net.eval()

    while(True):
        with torch.no_grad():
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 180, 480))) # 1024, 768 화면을 받아서 Numpy Array로 전환
            # screen = cv2.imread('./test_image2.jpg') # test image
            # input_image = Image.fromarray(screen)
            input_image = test_transform(image=screen)['image'].float().unsqueeze(0).to(device)
            # print(screen.float())
            # print(type(screen))
            # exit()
            # def test(model, test_loader, device):
            #     model.eval()
            #     correct = 0
            #     total = 0
            #     with torch.no_grad():
            #         for i, (image, labels) in enumerate(test_loader):
            #             image, labels = image.to(device), labels.to(device)
            #             output = model(image)
            #             _, argmax = torch.max(output, 1)
            #             total += image.size(0)
            #             correct += (labels == argmax).sum().item()
            #         acc = correct / total * 100
            #         print("acc for {} image: {:.2f}%".format(total, acc))
            #     model.train()
            #test_aug = test_transform(mode_flag="test")
            #test_dataset = screen, transform=test_aug
            #image, labels = image.to(device), labels.to(device)
            output = net(input_image)
            softmax_result = F.softmax(output)
            top_prob, top_label = torch.topk(softmax_result, 1)
            prob = round(top_prob.item() * 100, 2)
            label = labels.get(int(top_label))
            # print(f'prob: {prob}, label: {label}')

            Q = 0x10
            J = 0x24
            K = 0x25
            L = 0x26
            T = 0x14

            if (60 < prob) and (label == 'Q'):

                PressKey(Q)
                time.sleep(0.2)
                ReleaseKey(Q)



            elif (60 < prob) and (label == 'J'):
                PressKey(J)
                time.sleep(0.2)
                ReleaseKey(J)
            elif (90 < prob) and (label == 'K'):

                PressKey(K)
                #ReleaseKey(W)
                time.sleep(0.2)
                ReleaseKey(K)



            elif (60 < prob) and (label == 'L'):
                PressKey(L)

                time.sleep(0.2)
                ReleaseKey(L)





            # elif (50 < prob) and (label == 's'):
            #     PressKey(S)


            else:
                ReleaseKey(Q)
                ReleaseKey(J)
                ReleaseKey(K)
                ReleaseKey(L)


        print(prob, label)
    return prob, label


if __name__ == '__main__':
    predic_prob, predic_label = ingame_predic()