import os

import torch
import torch.nn as nn
from .utils import get_lr
import datetime
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def acc_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    # total_loss      = 0
    # total_accuracy  = 0
    # total_loss        =[]
    
    
    val_loss            = 0
    val_total_accuracy  = 0
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_step_val:
            break
        
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                
            optimizer.zero_grad()
            outputs = model_train(images)
            output  = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()

        if local_rank == 0:
            acc = val_total_accuracy / (iteration + 1)
            #total_loss.append(acc)
            # print(total_loss)
            # #画出准确概率曲线变化图 
            # plt.subplot(1, 1, 1)
            # plt.plot(acc, label='accurace')
            # plt.xlabel('Epoch')
            # plt.ylabel('acc')
            # plt.title('accurace')
            # plt.legend(loc="upper right")
            # plt.savefig("C:/Users/admin/Desktop/acc chart/epoch_acc_gailu.png")
            # plt.show()
            # plt.cla()
            # plt.close("all") 
    return acc