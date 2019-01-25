from getit2 import getJPG
from getit2 import gaussianThreshold
import cv2 as cv
import os
import random
import shutil
import numpy as np


def clearDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)


# # ------------------------- 获取所有图片 -------------------------
path = './13R0All/'
jpg_list = []
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.isdir(file_path):
        '''

        '''
        # if file in ['TSFIX', 'TSDFS', 'TGOTS']:
        #     continue
        jpg_list += getJPG(file_path, li=1)
print('一共有%d张照片' % len(jpg_list))
# 一共有16267张照片
# 一共有15420张照片

# # ------------------------- 读取分辨率 ------------------------
# fbv_list = []
# m = 0
# for jpg in jpg_list:
#     if m % 10 == 0:
#         print(m)
#     m += 1
#     jpg = cv.imread(os.path.join(jpg[0], jpg[1]))
#     if jpg.shape not in fbv_list:
#         fbv_list.append(jpg.shape)

# print(fbv_list)


# ------------------------- 随机抽取图片 -------------------------
# m = 0
# clearDir('./testp/')
# for i in jpg_list:
#     img = cv.imread(os.path.join(i[0], i[1]))
#     if random.randint(1, 100) < 11:
#         f_name = './testp/%s_%s' % (i[0][-5:], i[1])
#         shutil.copyfile(os.path.join(i[0], i[1]), f_name)
#         m += 1
#         print('Get %d, %s' % (m, f_name))


# # ------------------------- 测试高斯化 -------------------------
img = cv.imread(
    './testp/TTP2G_1300_TH8B0119AC_-1232.315_1283.284_1_10X_1_20181118202508.jpg')

img = gaussianThreshold(img, showimg=0)

cv.imwrite('./feature/binary.jpg', img)

# cv.waitKey(0)
# cv.destroyAllWindows()



'''
359 1078
'''
