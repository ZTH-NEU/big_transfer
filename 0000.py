import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import bit_pytorch.models as models
#
# args = './BiT-M-R50x1.npz'
#
# model = models.KNOWN_MODELS['BiT-M-R50x1'](head_size=10, zero_head=True)
# # model.load_from(np.load(args))
# num = 0
# for name ,param in model.named_parameters():
#     if num == 1 :
#         break
#     num += 1
#     print(name, '    ',param)
# # print(model)
import  argparse
import bit_common
parser = argparse.ArgumentParser(description="Fine-tune BiT-M model.")
parser.add_argument("--no-save", dest="save", action="store_false")
args = parser.parse_args()
if args.save:
    print('save')