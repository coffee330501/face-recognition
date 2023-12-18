import cv2 as cv
import os
from PIL import Image
import numpy as np


# 将人脸与FaceID绑定
def get_images_and_labels(path):
    # 存储人脸数据
    face_samples = []
    # 存储身份信息
    ids = []
    # 存储图片信息
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv.CascadeClassifier(
        'E:/WorkApplication/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    # 遍历图片
    for image_path in image_paths:
        # 打开图片，灰度化 PIL有九种模式： 1,L,P,RGB,RGBA,CMYK,YCbCr,I,F
        PIL_image = Image.open(image_path).convert('L')
        # 将图像转为数组
        img_numpy = np.array(PIL_image, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取图片ID和姓名
        id = int(os.path.split(image_path)[1].split('.')[0])
        # 预防无面容照片
        for (x, y, w, h) in faces:
            ids.append(id)
            face_samples.append(img_numpy[y:y + h, x:x + w])
        # 打印脸部特征和ID
        print('id:', id)
        print('fs:', face_samples)
    return face_samples, ids


if __name__ == '__main__':
    # 图片路径
    path = 'E:/Workspace/Python/FaceRecognition/img/cut'
    # 获取图像数组和id数组
    faces, ids = get_images_and_labels(path)
    # 加载识别器
    recognizer = cv.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(faces,np.array(ids))
    # 保存文件
    recognizer.write('trainer/trainer.yml')