import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    f = open('data/track2/test.txt', 'w')
    data = []
    for i in range(200):
        str_ = f'sample_{i}.jpg sample_{i}.jpg 518.8579\n'
        data.append(str_)
    f.writelines(data)