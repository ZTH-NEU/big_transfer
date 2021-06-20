# import torch
# import os
# print(torch.cuda.is_available())

# def mkdir(path):
#     # 引入模块
#     import os
#
#     # 去除首位空格
#     path = path.strip()
#     # 去除尾部 \ 符号
#     path = path.rstrip("\\")
#
#     # 判断路径是否存在
#     # 存在     True
#     # 不存在   False
#     isExists = os.path.exists(path)
#
#     # 判断结果
#     if not isExists:
#         # 如果不存在则创建目录
#         # 创建目录操作函数
#         os.makedirs(path)
#
#         print(path + ' 创建成功')
#         return True
#     else:
#         # 如果目录存在则不创建，并提示目录已存在
#         print(path + ' 目录已存在')
#         return False
#
#
# # 定义要创建的目录
# mkpath = ".\data"
# # 调用函数
# mkdir(mkpath)

import torch
import torch.utils.data as Data
from bit_pytorch import lbtoolbox as lb
from bit_pytorch.train import DogCat
import torchvision as tv
BATCH_SIZE = 1

def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
)
train_tx = tv.transforms.Compose([
    tv.transforms.Resize((160, 160)),
    tv.transforms.RandomCrop((80, 80)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
def show_batch():
    for epoch in range(3):
        for  batch_x , batch_y in recycle(loader):

            print("step:,batch_x:{}, batch_y:{}".format(batch_x,batch_y))



def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def show_batch1():
    # for epoch in range(3):
    step =0
    train_set = DogCat('G:/赵天祜/kaggle_DogsVSCats/train_mini/', transform=train_tx, train=True,test = False)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=10, shuffle=True,
        num_workers=1, pin_memory=True, drop_last=False)



    for x, y in recycle(train_loader):
        # training


        # print("steop:{}, batch_x:{}, batch_y：{}".format(step, batch_x, batch_y))
        step +=1
        print(step)

        # print(type(step))


if __name__ == '__main__':
    show_batch1()
    print("``````````````")
