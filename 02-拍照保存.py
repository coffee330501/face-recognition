import cv2 as cv;

cap = cv.VideoCapture(0)

num = 1
while cap.isOpened():
    ret, frame = cap.read()
    cv.imshow('frame', frame)
    # 按键判断
    k = cv.waitKey(1) & 0xFF
    # 保存
    if k == ord('s'):
        cv.imwrite('./img/cut/'+str(num)+'.jpg',frame)
        num+=1
    # 退出
    elif k == ord('q'):
        break

# 释放摄像头
cap.release()
# 释放内存
cv.destroyAllWindows()
