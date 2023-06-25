import cv2
import matplotlib.pyplot as plt
import numpy as np

root_path = 'data/test2/'


if __name__ == "__main__":
    for i in range(200):
        path1 = "sample_" + str(i) + ".jpg"
        path2 = str(i) + ".jpg"
        path3 = "gt" + str(i) + ".png"
        img1 = cv2.imread(root_path+path1)
        img2 = cv2.imread(root_path+path2)
        img3 = cv2.imread(root_path+path3)
        img3 = np.clip(img3*50, 0, 255)
        plt.subplot(131)
        plt.imshow(img1)
        plt.subplot(132)
        plt.imshow(img2)
        plt.subplot(133)
        plt.imshow(img3)
        plt.show()
