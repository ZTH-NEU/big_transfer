# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torch
import torchvision as tv

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
import bit_hyperrule
from bit_pytorch.dataset import DogCat
import os
# from scipy import interp
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score


def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i
  # for i in iterable:
  #   yield i


def mktrainval(args, logger):
  """Returns train and validation datasets."""
  # precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  precrop =160
  crop =128
  print('******************************************')
  print(precrop,crop)
  print('******************************************')

  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
  elif args.dataset == "zth":
    train_set = DogCat('E:/dogcat_data/train_mini/', transform=train_tx, train=True,test = False)
    valid_set = DogCat('E:/dogcat_data/test_mini/', transform=val_tx, train=False, test= True)
  elif args.dataset == "zth_mini":
    train_set = DogCat('E:/dogcat_data/2/', transform=train_tx, train=True, test=False)
    valid_set = DogCat('E:/dogcat_data/1/', transform=val_tx, train=False, test=True)
  elif args.dataset == "lung":
    train_set = DogCat('E:/data/train/', transform=train_tx, train=True,test = False)
    valid_set = DogCat('E:/data/test/', transform=val_tx, train=True, test= False)
  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")

  micro_batch_size = args.batch // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=True,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, chrono, logger, step):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()

  all_c, all_top1, all_top5 = [], [], []
  end = time.time()
  score_list = []  # 存储预测得分
  label_list = []  # 存储真实标签
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # measure data loading time
      chrono._done("eval load", time.time() - end)

      num_class = 2

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop"):
        logits = model(x)
        score_tmp = logits  # (batchsize, nclass)


        c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
        top1, top5 = topk(logits, y, ks=(1, 5))
        all_c.extend(c.cpu())  # Also ensures a sync point.
        all_top1.extend(top1.cpu())
        all_top5.extend(top5.cpu())

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(y.cpu().numpy())

    # measure elapsed time
    end = time.time()

  # 绘制ROC曲线
  score_array = np.array(score_list)
  score_array = score_array[:, :2]
  for i in range(len(score_array)):
    index_max = score_array[i].argmax()
    index_min = score_array[i].argmin()

    score_array[i][index_max] = 1
    score_array[i][index_min] = 0

  # 将label转换成onehot形式
  label_tensor = torch.tensor(label_list)
  label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
  label_onehot = torch.zeros(label_tensor.shape[0], num_class)
  label_onehot.scatter_(dim=1, index=label_tensor, value=1)
  label_onehot = label_onehot[:, :2]
  label_onehot = np.array(label_onehot)

  print("score_array:", score_array.shape)  # (batchsize, classnum)
  print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

  # 调用sklearn库，计算每个类别对应的fpr和tpr
  fpr_dict = dict()
  tpr_dict = dict()
  roc_auc_dict = dict()
  for i in range(num_class):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
  # micro
  fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
  roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

  # macro
  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(num_class):
    mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
  # Finally average it and compute AUC
  mean_tpr /= num_class
  fpr_dict["macro"] = all_fpr
  tpr_dict["macro"] = mean_tpr
  roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

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
  model.train()
  logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, "
              f"top5 {np.mean(all_top5):.2%}")
  logger.flush()
  return all_c, all_top1, all_top5


def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def main(args):
  logger = bit_common.setup_logger(args)

  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)

  logger.info(f"Loading model from {args.model}.npz")
  model = models.KNOWN_MODELS[args.model](head_size=10, zero_head=True)
  logger.info(f'head_size = valid_set.classes =10')
  model.load_from(np.load(f"{args.model}.npz"))

  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  step = 0

  # Note: no weight-decay!
  optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "bit.pth.tar")
  try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at step {step}")
  except FileNotFoundError:
    logger.info("Fine-tuning from BiT")

  model = model.to(device)
  optim.zero_grad()

  model.train()
  mixup = bit_hyperrule.get_mixup(len(train_set))
  cri = torch.nn.CrossEntropyLoss().to(device)

  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.time()

  with lb.Uninterrupt() as u:
    for x, y in recycle(train_loader):
      # measure data loading time, which is spent in the `for` statement.
      chrono._done("load", time.time() - end)

      if u.interrupted:
        break

      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # Update learning-rate, including stop training if over.
      lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr

      if mixup > 0.0:
        x, y_a, y_b = mixup_data(x, y, mixup_l)


        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)


      # compute output
      with chrono.measure("fprop"):
        logits = model(x)

        if mixup > 0.0:
          c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
        else:
          c = cri(logits, y)
        c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.


      # Accumulate grads
      with chrono.measure("grads"):
        (c / args.batch_split).backward()
        accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
      logger.flush()

      # Update params
      if accum_steps == args.batch_split:
        with chrono.measure("update"):
          optim.step()
          optim.zero_grad()
        step += 1
        accum_steps = 0
        # Sample new mixup ratio for next batch
        mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          run_eval(model, valid_loader, device, chrono, logger, step)

      end = time.time()
    if args.save:
      torch.save({
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
      }, savename)
      logger.info('f Running save.....')
    # Final eval at end of training.



    run_eval(model, valid_loader, device, chrono, logger, step='end')

  logger.info(f"Timings:\n{chrono}")


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  main(parser.parse_args())
