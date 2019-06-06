import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


video_name = "video.mp4"

def process():
    print("Iniciando reconhecimento de dados")
    cap = cv2.VideoCapture(video_name)
    while True:
        _, frame = cap.read()
        frame = find_dice(frame)
        cv2.imshow('Dice video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def treat(frame):
    gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    kernel = np.ones((13, 13), np.uint8)
    dilated_frame = cv2.dilate(gray_frame, kernel)
    return cv2.erode(dilated_frame, kernel)


def find_contours(image):
    _,thresh = cv2.threshold(image, 245, 250, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def is_approx_valid(contour):
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    if len(approx) >= 6 and len(approx) <= 12:
        return True
    return False


def check_faces(contour, image, original):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img_crop, img_rot = crop_rect(original, rect)
    return find_faces(img_crop)


def find_dice(frame):
    treated_image = treat(frame)
    for contour in find_contours(treated_image):
        if is_approx_valid(contour):
            (x, y, w, h) = cv2.boundingRect(contour)
            face = check_faces(contour, treated_image, frame)
            cv2.putText(frame, face, (x + int(w / 2), y + int(h / 2)), cv2.FONT_HERSHEY_TRIPLEX, 1, (0))
    return frame


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def check_offset(height, width, offset):
    max_offset = 200
    min_offset = 30
    if height > offset + 4 and width > offset + 4 \
        and height < max_offset and width < max_offset \
        and height > min_offset and width > min_offset:
            return True
    return False


def find_faces(image):
    height, width = image.shape[0], image.shape[1]
    offset = 3
    if check_offset(height, width, offset):
        image_copy = image[offset:width-offset, offset:height-offset].copy()
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 250, cv2.THRESH_BINARY_INV)
        kernel = np.ones((4, 4), np.uint8)
        dilated_image = cv2.dilate(thresh, kernel)
        eroded_image = cv2.erode(dilated_image, kernel)
        contours, hierarchy = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(gray, [approx], 0, (0), 5)
            if len(approx) >= 5:
                dots += 1
        return str(dots)


if __name__ == "__main__":
    process()

    cv2.waitKey()
    cv2.destroyAllWindows()