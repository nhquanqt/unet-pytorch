import os, time, sys

import torch
import torchvision.transforms.functional as TF

from PIL import Image

def main():
    root = '/home/wan/dataset/cityscapes'
    goal = 'dataset/cityscapes'

    for split in ['train', 'val']:
        images_dir = os.path.join(root, 'leftImg8bit', split)
        targets_dir = os.path.join(root, 'gtFine', split)

        os.makedirs(os.path.join(goal, 'leftImg8bit', split))
        os.makedirs(os.path.join(goal, 'gtFine', split))

        for city in os.listdir(images_dir):
            print(city)
            os.makedirs(os.path.join(goal, 'leftImg8bit', split, city))
            os.makedirs(os.path.join(goal, 'gtFine', split, city))

            img_dir = os.path.join(images_dir, city)
            target_dir = os.path.join(targets_dir, city)

            cnt = 0

            for file_name in os.listdir(img_dir):
                image = Image.open(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'gtFine_labelIds.png')
                target = Image.open(os.path.join(target_dir, target_name))

                image = TF.resize(image, 256, interpolation=Image.BILINEAR)
                target = TF.resize(target, 256, interpolation=Image.NEAREST)

                image.save(os.path.join(goal, 'leftImg8bit', split, city, file_name))
                target.save(os.path.join(goal, 'gtFine', split, city, target_name))

                cnt += 1
                if cnt % 100 == 0:
                    print(cnt)
            
            print(cnt)

if __name__ == "__main__":
    main()