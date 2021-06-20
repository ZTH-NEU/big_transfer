from sklearn.model_selection import train_test_split
import os
from shutil import copy
from sys import exit
#data:需要进行分割的数据集
#random_state:设置随机种子，保证每次运行生成相同的随机数
#test_size:将数据分割成训练集的比例
root = 'E:/dogcat_data/train/'
imgs = [os.path.join(root, img) for img in os.listdir(root)]
train_set, test_set = train_test_split(imgs, test_size=0.2, random_state=42)
print(len(train_set),len(test_set))
destination_file = 'E:/dogcat_data/train_mini/'
destination_file2 = 'E:/dogcat_data/test_mini/'
for i in train_set:
    source_file = i
    copy(source_file, destination_file )
for j in test_set:
    source_file = j
    copy(source_file, destination_file2)
