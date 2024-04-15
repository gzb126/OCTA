import cv2
import numpy as np
import os
from PIL import Image
from scipy import ndimage
import shutil
from skimage import morphology
from hessian import FAZ_Preprocess
from OCTNet import OCTANetwork
import math
import csv


def ChooseCircle(gray, angbegin, angend):
    h, w = gray.shape
    sp = (w // 2, h // 2)
    Mask = np.zeros((h, w), np.uint8) * 255.0
    mask = np.zeros((h, w), np.uint8) * 255.0
    # 参数 1.目标图片  2.椭圆圆心  3.长短轴长度  4.偏转角度  5.圆弧起始角度  6.终止角度  7.颜色  8.是否填充
    cv2.ellipse(Mask, sp, (int(min(h, w) // 2), int(min(h, w) // 2)), 0, angbegin, angend, (255, 255, 255), -1)
    cv2.circle(mask, sp, int(min(h, w) // 6), (255, 255, 255), -1)  # 内圆
    return Mask - mask

def ChooseCircleCenter(gray):
    h, w = gray.shape
    sp = (w // 2, h // 2)
    Mask = np.zeros((h, w), np.uint8) * 255.0
    cv2.circle(Mask, sp, int(min(h, w) // 6), (255, 255, 255), -1)  # 内圆
    return Mask

def ComputeEveryOne(gray, area):

    area[area>0] = 255
    area[area<0] = 0
    contours, _ = cv2.findContours(area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    area = area/255
    ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    whit_area = cv2.countNonZero(area)
    choseMat = binary * area
    size_elements = cv2.countNonZero(choseMat)

    conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(conts)):  # 拆除交点后进行预支判定，小于阈值的就不要
        are = cv2.contourArea(conts[j])
        if are < 10:
            cv2.fillPoly(binary, [conts[j]], 0)
    skeleton0 = morphology.skeletonize(binary, method="lee")  # 细化提取骨架
    intMat = skeleton0.astype(np.uint8)
    chosearea = intMat * area
    Length = cv2.countNonZero(chosearea)

    VD = round((3*Length/gray.shape[1]) / (3*3*whit_area/(gray.shape[0]*gray.shape[1])), 2)  # 血管长度/区域面积
    PD = round(size_elements / whit_area, 2)
    VDI = round((3*3*size_elements/(gray.shape[0]*gray.shape[1])) / (3*Length/gray.shape[1]), 3)

    return VD, PD, VDI, (cx, cy)


def ComputeCenterOne(gray, Mask, area):

    area[area>0] = 255
    area[area<0] = 0
    contours, _ = cv2.findContours(area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    area = area/255
    ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    whit_area = cv2.countNonZero(area)
    choseMat = binary * area
    size_elements = cv2.countNonZero(choseMat)

    bina = binary.copy()
    intersection = cv2.bitwise_and(bina, Mask)
    bina = bina - intersection

    conts, _ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(conts)):  # 拆除交点后进行预支判定，小于阈值的就不要
        are = cv2.contourArea(conts[j])
        if are < 10:
            cv2.fillPoly(bina, [conts[j]], 0)
    skeleton0 = morphology.skeletonize(bina, method="lee")  # 细化提取骨架
    intMat = skeleton0.astype(np.uint8)
    chosearea = intMat * area
    Length = cv2.countNonZero(chosearea)

    VD = round((3*Length/gray.shape[1]) / (3*3*whit_area/(gray.shape[0]*gray.shape[1])), 2)  # 血管长度/区域面积
    PD = round(size_elements / whit_area, 2)
    VDI = round((3*3*size_elements/(gray.shape[0]*gray.shape[1])) / (3*Length/gray.shape[1]), 3)

    return VD, PD, VDI, (cx, cy)

def DrawImg(cv_img):
    h, w, c = cv_img.shape
    cv2.circle(cv_img, (w // 2, h // 2), int(min(w // 2, h // 2)), (255, 255, 255), 2)  # 外圆
    cv2.circle(cv_img, (w // 2, h // 2), int(min(w // 6, h // 6)), (255, 255, 255), 2)  # 内圆

    P1 = (int((h-1.414*0.5*h)/2), int((h-1.414*0.5*h)/2))
    p1 = (int(h/2 - h/(6*1.414)), int(h/2 - h/(6*1.414)))
    cv2.line(cv_img, P1, p1, (255,255,255), 2)

    P2 = (h - int((h - 1.414 * 0.5 * h)/2), int((h-1.414*0.5*h)/2))
    p2 = (int(h/2 + h/(6 * 1.414)), int(h/2 - h/(6*1.414)))
    cv2.line(cv_img, P2, p2, (255, 255, 255), 2)

    P3 = (h - int((h - 1.414 * 0.5 * h)/2), h -int((h - 1.414 * 0.5 * h)/2))
    p3 = (int(h/2 + h/(6 * 1.414)), int(h/2 + h/(6 * 1.414)))
    cv2.line(cv_img, P3, p3, (255, 255, 255), 2)

    P4 = (int((h-1.414*0.5*h)/2), h - int((h - 1.414 * 0.5 * h) / 2))
    p4 = (int(h/2 - h/(6*1.414)), int(h / 2 + h / (6 * 1.414)))
    cv2.line(cv_img, P4, p4, (255, 255, 255), 2)

    return cv_img


def CsvWrite(fil, count):
    f = open(fil, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(count)
    f.close()

def main():
    net = OCTANetwork('./models/Se_resnext50-920eef84.pth', 'Se_resnext50')
    root = r'F:\2_PycharmWorks\OCTAReadProject\imgfiles\33'
    head = ['图像name', 'VD_S1',	'VD_N1', 'VD_L1', 'VD_T1', 'VD_C0', 'PD_S1', 'PD_N1', 'PD_L1', 'PD_T1',
            'PD_C0', 'VDI_S1', 'VDI_N1', 'VDI_L1', 'VDI_T1', 'VDI_C0', 'FAZ面积', 'FAZ周长', 'FAZ形态指数',
            'VD均值', 'PD均值', 'VDI均值']
    CsvWrite(r'F:\2_PycharmWorks\OCTAReadProject\imgfiles\out/' + 'result.csv', head)
    imgs = os.listdir(root)
    for img in imgs:
        path = os.path.join(root, img)
        Mask = net.forword(path)
        cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        Mask = cv2.resize(Mask, gray.shape)

        image = FAZ_Preprocess(path, [0.5, 1, 1.5, 2, 2.5], 0.5, 0.5)
        image = image.vesselness2d()
        image = image * 255
        image = image.astype(np.uint8)

        # show = cv2.hconcat([gray, image])
        # cv2.imencode('.jpg', show)[1].tofile(r'F:\2_PycharmWorks\OCTAReadProject\imgfiles\out/' + img)

        L1 = ChooseCircle(gray, 45, 135)
        T1 = ChooseCircle(gray, 135, 225)
        S1 = ChooseCircle(gray, 225, 315)
        N1 = ChooseCircle(gray, 0, 45) + ChooseCircle(gray, 315, 360)
        C1 = ChooseCircleCenter(gray)

        VDS, PDS, VDIS, pS = ComputeEveryOne(image, S1)
        VDN, PDN, VDIN, pN = ComputeEveryOne(image, N1)
        VDL, PDL, VDIL, pL = ComputeEveryOne(image, L1)
        VDT, PDT, VDIT, pT = ComputeEveryOne(image, T1)
        CVD, CPD, CVDI, Cp = ComputeCenterOne(image, Mask, C1)

        im = DrawImg(cv_img)
        cv2.putText(im, str(VDL) + ' ' + str(PDL) + ' ' + str(VDIL), (pL[0]-100, pL[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(im, str(VDT) + ' ' + str(PDT) + ' ' + str(VDIT), (pT[0]-100, pT[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(im, str(VDS) + ' ' + str(PDS) + ' ' + str(VDIS), (pS[0]-100, pS[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(im, str(VDN) + ' ' + str(PDN) + ' ' + str(VDIN), (pN[0]-100, pN[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(im, str(CVD) + ' ' + str(CPD) + ' ' + str(CVDI), (Cp[0]-100, Cp[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        binary = Mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        zeros = np.zeros((im.shape), dtype=np.uint8)
        points = np.array([contours[max_id]], dtype=np.int32)
        ma = cv2.fillPoly(zeros, points, color=(80, 127, 255))
        mask_img = 0.4 * ma + im

        area = cv2.contourArea(contours[max_id])
        area_mm = round(3*3*area/(gray.shape[0]*gray.shape[1]), 2)
        perimeter = cv2.arcLength(contours[max_id], True)  # 轮廓周长 (perimeter)
        circular = round(4 * np.pi * area / (perimeter ** 2), 2)  # 轮廓的圆度 (circularity)
        per = round(3*perimeter/gray.shape[1], 2)

        cv2.putText(mask_img, str(area_mm) + 'mm2' + ' ' + str(per) + 'mm '+ str(circular), (gray.shape[0]//2-100, gray.shape[1]//2-100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
        cv2.imencode('.jpg', mask_img)[1].tofile(r'F:\2_PycharmWorks\OCTAReadProject\imgfiles\out/' + img)

        Y = [img, str(VDS), str(VDN), str(VDL), str(VDT), str(CVD), str(PDS), str(PDN), str(PDL), str(PDT), str(CPD),
             str(VDIS), str(VDIN), str(VDIL), str(VDIT), str(CVDI), str(area_mm), str(per), str(circular),
             str(np.mean((VDS, VDN, VDL, VDT, CVD))), str(np.mean((PDS, PDN, PDL, PDT, CPD))), str(np.mean((VDIS, VDIN, VDIL, VDIT, CVDI)))]
        CsvWrite(r'F:\2_PycharmWorks\OCTAReadProject\imgfiles\out/' + 'result.csv', Y)


if __name__ == '__main__':
    main()
