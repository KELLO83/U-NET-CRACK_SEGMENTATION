from crackseg.utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

if __name__ == "__main__":
    image_path = 'data/CRKWH100_IMAGE/test'
    print("origin :",image_path)
    test = image_path.split('/')
    target = test[1]
    target_ = target.find('_')
    target__ = target[:target_+1] + 'MASK'
    res = test[0] , target__ , test[2]
    res__ = '/'.join(res)
    

    print(target__)
    print(test)
    print(res)
    print(res__)
    
    mage_path = 'data/CRKWH100_IMAGE/test'
    print("origin :", image_path)

    # 이미지 경로를 '/'로 분리
    parts = image_path.split('/')

    # 목표 문자열 조작
    target = parts[1]
    mask_target = target.split('_')[0] + '_MASK'

    # 결과 문자열 생성
    res = f"{parts[0]}/{mask_target}/{parts[2]}"

    print(mask_target)
    print(parts)
    print((parts[0], mask_target, parts[2]))
    print(res)
    
    