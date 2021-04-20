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

# import torch
# import torch.utils.data as Data
#
# BATCH_SIZE = 5
#
# x = torch.linspace(1, 10, 10)
# y = torch.linspace(10, 1, 10)
# # 把数据放在数据库中
# torch_dataset = Data.TensorDataset(x, y)
# loader = Data.DataLoader(
#     # 从数据库中每次抽出batch size个样本
#     dataset=torch_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=2,
# )
#
# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
#
# train_ids = Data.TensorDataset(a, b) #相当于zip函数
#
# # 切片输出
#
# print(train_ids[0:1])
#
# print('=' * 60)
#
# # 循环取数据
#
# for x_train, y_label in train_ids:
#
#     print(x_train, y_label)
# def show_batch():
#     for epoch in range(3):
#         for  batch_x , batch_y in recycle(loader):
#             # training
#
#             print("batch_x:{}, batch_y:{}".format(batch_x,batch_y))
#
#             x, y_a, y_b =mixup_data(batch_x,batch_y,0.1)
#             print(x,y_a,y_b)
#
#
# def recycle(iterable):
#     """Variant of itertools.cycle that does not save iterates."""
#     # while True:
#     for i in iterable:
#         yield i
#
#
# def show_batch1():
#     for epoch in range(3):
#         for step, (batch_x, batch_y) in enumerate(loader):
#             # training
#
#             print("steop:{}, batch_x:{}, batch_y：{}".format(step, batch_x, batch_y))
#             print(type(step))
#
#
#
#
# def mixup_data(x, y, l):
#     """Returns mixed inputs, pairs of targets, and lambda"""
#     indices = torch.randperm(x.shape[0]).to(x.device)
#     print(indices)
#
#     mixed_x = l * x + (1 - l) * x[indices]
#     y_a, y_b = y, y[indices]
#     return mixed_x, y_a, y_b
#
#
# if __name__ == '__main__':
#     show_batch()
#     print("``````````````")

# import numpy as np
# import torch
# mixup = 0.1
# mixup_l = np.random.beta(mixup, mixup,10) if mixup > 0 else 1
# print(mixup_l)
# x = 10
# indices = torch.randperm(x.shape[0]).to(x.device)
# print(indices)
# a = 15000
# if a < 10_000:
#     print("yes")
cout = None
cin = 10
cmid = None
cout = cout or cin
cmid = cmid or cout // 4
print(cout)
print(cmid)