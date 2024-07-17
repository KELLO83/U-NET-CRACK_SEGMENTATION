import cv2
import numpy as np

def resize_and_pad(image, size=(512, 512)):
    h, w = image.shape[:2]
    scale = size[0] / max(h, w)
    
    # 비율 유지 리사이징
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # 패딩 추가
    delta_w = size[1] - new_w
    delta_h = size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]  # 패딩 색상 (검정색)
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    print("resize image :",padded_image.shape)
    return padded_image

# 이미지 로드
image1 = cv2.imread('11.jpg')  # 960x720 이미지
image2 = cv2.imread('6194.jpg')  # 800x600 이미지

cv2.imshow('image1',image1)
cv2.imshow('image2',image2)
if image1 is None or image2 is None:
    assert f"file {image1} {image2} not exist"
    
# 리사이징 및 패딩 적용
image1_resized = resize_and_pad(image1)
image2_resized = resize_and_pad(image2)

# 결과 이미지 보기
cv2.imshow('Image 1 Resized and Padded', image1_resized)
cv2.imshow('Image 2 Resized and Padded', image2_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
