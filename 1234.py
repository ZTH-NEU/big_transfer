import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from collections import Counter

import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

test = [0,1,0,1,0,1]
print(Counter(test))
label_list = [0,1,0,1,1,1]
score_array = np.array([[1,0,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0]
                        ])
print(score_array)
num_class = 2

label_tensor = torch.tensor(label_list)
label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
label_onehot = torch.zeros(label_tensor.shape[0], 10)
label_onehot.scatter_(dim=1, index=label_tensor, value=1)
label_onehot = np.array(label_onehot)

print(label_onehot)


fpr_dict = dict()
tpr_dict = dict()
roc_auc_dict = dict()
for i in range(num_class):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
print(fpr_dict)
print(tpr_dict)


# micro
fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

# macro
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
print(all_fpr)
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_class):
    mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

# Finally average it and compute AUC
mean_tpr /= num_class
fpr_dict["macro"] = all_fpr
tpr_dict["macro"] = mean_tpr
roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
print("score_array:", score_array.shape)  # (batchsize, classnum)
print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])
# 绘制所有类别平均的roc曲线
plt.figure()
lw = 2
plt.plot(fpr_dict["micro"], tpr_dict["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_dict["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr_dict["macro"], tpr_dict["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_dict["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_class), colors):
    plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc_dict[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('set113_roc.jpg')
plt.show()

