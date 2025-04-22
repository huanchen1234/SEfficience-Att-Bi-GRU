import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

#from train import acc1


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        self.acc=[]
        self.acc1=[]

        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.acc.append(1-self.losses[len(self.losses)-1])
        self.acc1.append(1-self.val_loss[len(self.losses)-1])

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_test_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('test_loss', val_loss, epoch)
        self.loss_plot()
        #self.acc_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='test loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth test loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")
# #---------------------------------
# #计算准确率
# #------------------------------
#     def acc_plot(self):
    
#         iters1 = range(len(self.losses))
#         plt.figure()
#         plt.plot(iters1, acc1, 'red', linewidth = 2, label='train acc')
#         #plt.plot(iters1, self.acc1, 'coral', linewidth = 2, label='val acc')
#         try:
#             if len(self.losses) < 25:
#                 num = 5
#             else:
#                 num = 15
            
#             plt.plot(iters1, scipy.signal.savgol_filter(self.acc, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
#             #plt.plot(iters1, scipy.signal.savgol_filter(self.acc1, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
#         except:
#             pass 

#         plt.grid(True)
#         plt.xlabel('Epoch')
#         plt.ylabel('acc')
#         plt.legend(loc="upper right")

#         plt.savefig("C:/Users/admin/Desktop/acc chart/epoch_acc.png")

#         plt.cla()
#         plt.close("all")

        
