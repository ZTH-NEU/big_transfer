import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
a ='E:/data/train/1-baojinchun-8_644BF14F8B1740C190FBDF7EE12CEB36-70-0.png'

img = plt.imread(a)
img2  =  Image.open(a).convert('RGB')
#图片的高H为460，宽W为346，颜色通道C为3
print(img.shape)
print(img.dtype)
print(type(img))

img2 = np.array(img2)
#
# print(img2.shape)
# print(img2.dtype)
print(type(img2))

plt.imshow(img)
plt.show()

plt.imshow(img2)
plt.show()

a ='E:/data/train/1-baojinchun-8_644BF14F8B1740C190FBDF7EE12CEB36-70-0.png'

b = int(a.split('.')[-2].split('-')[-1])

print(b)

str = 'python -m bit_pytorch.train --name cifar10_date_lung_  --model BiT-M-R50x1 --logdir /tmp/bit_logs --dataset lung --datadir data  --batch 512 --batch_split 2'
