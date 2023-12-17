import cv2 as cv;

def face_recognition(img):
    # 灰度转换
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 分类器（别人训练好的模型） 在opencv安装路径 opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml
    face_rec = cv.CascadeClassifier('E:/WorkApplication/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    # 识别人脸位置 arg0: 图片 arg1: 每次遍历之后的缩放倍数,减小可以提高速度(有时也可以提高精度) arg2: 检测X次后仍有人脸才返回真 arg3: 填0就行 minSize: 人脸最小范围 maxSize: 人脸最大范围
    # face = face_rec.detectMultiScale(gray_img,1.1,5,0,(10,10),(100,100))
    face = face_rec.detectMultiScale(gray_img)
    # 坐标
    for (x,y,w,h) in face:
        # 绘制矩形 arg0: 图片 arg1: [x轴起始点,y轴起始点,宽,高] color: [B,G,R] thickness: 宽度
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
    # 显示图片
    cv.imshow('face',img)


# 读取摄像头 0:默认摄像头 其他的为外接摄像头
# cap = cv.VideoCapture(0)
# 读取本地视频，也可以读取链接
cap = cv.VideoCapture('video/face.mp4')

# 等待 0:无限等待
while True:
    # res0: 即self,返回值是否有内容 res1: 返回帧
    flag,frame = cap.read()
    # 播放完毕
    if not flag:
        break
    face_recognition(frame)
    # 读取摄像头需要设延时时间不然会卡住
    cv.waitKey(10)
# 释放内存
cv.destroyAllWindows()
