import os
import cv2
import numpy as np


def match(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    loss = img1 - img2
    cv2.imshow('1', loss)
    cv2.waitKey(0)
    loss = np.clip(loss, 0, 255)
    cv2.imshow('2', loss)
    cv2.waitKey(0)
    ans = 0
    for i in range(loss.shape[0]):
        for j in range(loss.shape[1]):
            ans += ((loss[i][j][0]*0.299+loss[i][j][1]*0.587+loss[i][j][2]*0.114)/255) ** 2
    return ans


if __name__ == "__main__":
    img1 = cv2.imread('data/nyu/basement_0001a/sync_depth_00000.png')
    print(img1.shape)
    print(img1)

    match('data/track2\sample_0.jpg', 'data/nyu/bathroom/rgb_00731.jpg')
    exit()

    ans_list = []

    competition_path = 'data/track2'
    nyu_path = 'data/nyu/bathroom'

    competition_list = os.listdir(competition_path)
    for competition_file in competition_list:
        if not competition_file.endswith('.jpg'):
            continue
        min_loss = 999999
        for root, dirs, files in os.walk(nyu_path):
            for file in files:
                if not file.endswith('.jpg'):
                    continue
                path1 = os.path.join(competition_path, competition_file)
                path2 = os.path.join(root, file)
                temp = match(path1, path2)
                if temp < min_loss:
                    min_loss = temp
                    print(path1, path2)
                    print(min_loss)
                    print("--------------------------------------------------------")
        exit()
