import torch
from torchvision.io import image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import imageio
import os

root_path = "G:/PyCharmWorksapce/Monocular-Depth-Estimation-Toolbox-main/"
threshold = 20


class DepthLevelMix():
    def __init__(self):
        self.datapath = root_path + "data/mixed_dataset/"
        self.originaldatapath = root_path + "data/nyu/"

    def levelmix_floder(self, folder_path1: str, folder_path2: str) -> None:
        # 1. create a new folder
        folder_path_new = folder_path1 + "-" + folder_path2
        folder_path_new = self.datapath + folder_path_new
        if not os.path.exists(folder_path_new):
            os.mkdir(folder_path_new)
            print("create floder with name is " + folder_path1 + "-" + folder_path2)
        else:
            print("folder exists")

        # 2.find two pictures and levelmix
        for root1, dirs1, files1 in os.walk(self.originaldatapath + folder_path1, topdown=False):
            for root2, dirs2, files2 in os.walk(self.originaldatapath + folder_path2, topdown=False):
                for i, name1 in enumerate(files1):
                    if name1.startswith("rgb"):
                        ip1 = self.originaldatapath + folder_path1 + "/" + name1
                        dp1 = self.originaldatapath + folder_path1 + "/sync_depth" + str(name1[3:-3] + "png")
                    else:
                        continue
                    for j, name2 in enumerate(files2):
                        if name2.startswith("rgb"):
                            ip2 = self.originaldatapath + folder_path2 + "/" + name2
                            dp2 = self.originaldatapath + folder_path2 + "/sync_depth" + str(name1[3:-3] + "png")
                        else:
                            continue
                        print(ip1, dp1, ip2, dp2)
                        img1_ans, depth1_ans, img2_ans, depth2_ans = self.levelmix(ip1, dp1, ip2, dp2)
                        # 保存图片和深度图
                        print(folder_path_new + f'rgb_{i}-{j}.jpg')
                        imageio.imwrite(folder_path_new + f'/rgb_{i}-{j}.jpg', img1_ans)
                        imageio.imwrite(folder_path_new + f'/sync_depth_{i}-{j}.jpg', depth1_ans)
                        imageio.imwrite(folder_path_new + f'/rgb_{j}-{i}.jpg', img2_ans)
                        imageio.imwrite(folder_path_new + f'/sync_depth_{j}-{i}.jpg', depth2_ans)

                        # 记录在train文件中
                        # Note = open(root_path + 'data\mixed_dataset\train.txt', mode='w')
                        # Note.write(folder_path_new + f'/rgb_{i}-{j}.jpg ')
                        # Note.write(folder_path_new + f'/sync_depth_{i}-{j}.jpg' + "518.8579\n")
                        # Note.write(folder_path_new + f'/rgb_{j}-{i}.jpg ')
                        # Note.write(folder_path_new + f'/sync_depth_{j}-{i}.jpg' + "518.8579\n")
                        # Note.close()
                        print(i, j)
                        break
                    print(i, j)
                    break


    def levelmix(self, img_path1, depth_path1, img_path2, depth_path2):
        img1 = cv.imread(img_path1)
        img1 = cv.cvtColor(img1, code=cv.COLOR_BGR2RGB)

        depth1 = cv.imread(depth_path1)
        depth1 = cv.cvtColor(depth1, code=cv.COLOR_BGR2RGB)

        img2 = cv.imread(img_path2)
        img2 = cv.cvtColor(img2, code=cv.COLOR_BGR2RGB)

        depth2 = cv.imread(depth_path2)
        depth2 = cv.cvtColor(depth2, code=cv.COLOR_BGR2RGB)

        img1_ans = np.array(img1)
        img2_ans = np.array(img2)
        depth1_ans = np.array(depth1)
        depth2_ans = np.array(depth2)

        # plt.imshow(img1) # 显示图片
        # plt.show()
        # plt.imshow(img2) # 显示图片
        # plt.show()

        if depth1.shape == depth2.shape:
            print("Two pictures have same shape")
            cnt = 0
            for i in range(depth1.shape[0]):
                for j in range(depth1.shape[1]):
                    if depth1[i][j][0] >= threshold:
                        cnt += 1
                        img2_ans[i][j] = img1[i][j]
                        depth2_ans[i][j] = depth1[i][j]
            print(cnt / depth1.shape[0] / depth1.shape[1])
            cnt = 0
            for i in range(depth2.shape[0]):
                for j in range(depth2.shape[1]):
                    if depth2[i][j][0] >= threshold:
                        cnt += 1
                        img1_ans[i][j] = img2[i][j]
                        depth1_ans[i][j] = depth2[i][j]
            print(cnt / depth2.shape[0] / depth2.shape[1])
        else:
            print("Two pictures do not have same shape")

        # plt.imshow(img1_ans) # 显示图片
        # plt.show()
        # plt.imshow(img2_ans) # 显示图片
        # plt.show()

        return img1_ans, depth1_ans, img2_ans, depth2_ans


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # x = DepthLevelMix()
    # x.levelmix_floder(r"basement_0001a", r"basement_0001b")

