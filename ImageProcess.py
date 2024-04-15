import cv2
import os
import  numpy as np
from skimage import morphology


def ImgProcess(root):
    imgs = os.listdir(root)
    for img in imgs:
        path = os.path.join(root, img)
        cv_img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)

        # alpha = 1.5  # 对比度控制
        # beta = 1  # 亮度控制
        # cv_img = cv2.convertScaleAbs(cv_img, alpha=alpha, beta=beta)
        # show = cv2.hconcat([cv_img, cv_img])
        # cv2.imshow('adjusted', show)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret, binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

        size_elements = 0
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i][j] == 255:
                    size_elements += 1


        skeleton0 = morphology.skeletonize(binary, method="lee")  # 细化提取骨架
        intMat = skeleton0.astype(np.uint8)

        Length = 0

        for i in range(intMat.shape[0]):
            for j in range(intMat.shape[1]):
                if intMat[i][j] == 255:
                    cv_img[i][j] = [0, 0, 255]
                    Length += 1

        # cv2.imencode('.jpg', cv2.hconcat([img_gray, binary, intMat]))[1].tofile('./imgfiles/out/' + img)
        cv2.imencode('.jpg', cv_img)[1].tofile('./imgfiles/out/' + img)
        print(img)
        print(round(Length/(intMat.shape[0]*intMat.shape[1]), 2))
        print(round(size_elements/(intMat.shape[0]*intMat.shape[1]), 2))

        # cv2.imencode('.jpg', binary2)[1].tofile('./imgfiles/out/' + img)


def ImgMakeKD(root):
    imgs = os.listdir(root)
    for img in imgs:
        path = os.path.join(root, img)
        cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        h, w = img_gray.shape
        zp = (w//2, h//2)
        w_cup = 600
        img_cup = img_gray[int(zp[1] - w_cup/2):int(zp[1] + w_cup/2), int(zp[0] - w_cup/2):int(zp[0] + w_cup/2)]

        # cv2.imencode('.jpg', img_cup)[1].tofile(r'F:\1_PycharmProjects\U-2-Net-master\test_data/' + img)

        ret, binary = cv2.threshold(img_cup, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # ret, binary = cv2.threshold(img_cup, 40, 255, cv2.THRESH_BINARY)
        Mat = binary.copy()
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for j in range(len(contours)):   # 拆除交点后进行预支判定，小于阈值的就不要
            area = cv2.contourArea(contours[j])
            if area < 10:
                cv2.fillPoly(Mat, [contours[j]], 0)
        kernel = np.ones((4, 4), np.uint8)
        Mat = cv2.dilate(Mat, kernel, iterations=3)
        show = cv2.hconcat([img_cup, binary, Mat])
        cv2.imencode('.jpg', show)[1].tofile(r'./imgfiles\out/' + img)


if __name__ == '__main__':
    root = r'./imgfiles/33'
    # ImgProcess(root)
    ImgMakeKD(root)