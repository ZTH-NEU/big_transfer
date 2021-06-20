# coding:utf8
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):

    def __init__(self, root, transform=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        self.transform = transform
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg 
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('-')[0].split('/')[-1]))

            # imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            # self.imgs = imgs[:int(0.7 * imgs_num)]
            self.imgs = imgs
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transform is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = int(img_path.split('.')[-2].split('-')[-1])
            # label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data_pre = Image.open(img_path).convert('RGB')
        data = self.transform(data_pre)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    train_data = DogCat('D:/BaiduNetdiskDownload/kaggle_DogsVSCats/train_mini/', train=True)
    picture, label = train_data.__getitem__(0)
    picture = picture.permute(1, 2, 0)

    # picture = picture.reshape((224, 224, 3))
    plt.imshow(picture)
    plt.show()
    print(label)
