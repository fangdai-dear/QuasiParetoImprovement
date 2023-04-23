import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

def Divide(outputs, labels, subggroup):
    Min_img, Min_lab, Maj_img, Maj_lab = torch.ones(0, len(labels[0])), torch.ones(0, len(labels[0])), \
                                        torch.ones(0, len(labels[0])), torch.ones(0, len(labels[0]))
    for x in range(len(subggroup)):
        if subggroup[x].tolist()[0] == 1 :
            Min_img = torch.cat((Min_img, outputs[x].detach().unsqueeze(0)), 0)
            Min_lab = torch.cat((Min_lab, labels[x].detach().unsqueeze(0)), 0)
        else:
            Maj_img = torch.cat((Maj_img, outputs[x].detach().unsqueeze(0)), 0)
            Maj_lab = torch.cat((Maj_lab, labels[x].detach().unsqueeze(0)), 0)
    return Maj_img, Maj_lab, Min_img, Min_lab

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, subggroup, outputs):
        Min_img, Maj_img, = torch.ones(0, 100), torch.ones(0, 100)
        for x in range(len(subggroup)):
            if subggroup[x].tolist()[0] == 1 :
                Min_img = torch.cat((Min_img, outputs[x].detach().unsqueeze(0)), 0)
            else:
                Maj_img = torch.cat((Maj_img, outputs[x].detach().unsqueeze(0)), 0)

        source = Min_img
        target = Maj_img

        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class BatchSpectralShrinkage(nn.Module):

    def __init__(self, k=1):
        super(BatchSpectralShrinkage, self).__init__()
        self.k = k

    def forward(self, feature):
        result = 0
        u, s, v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.k):
            result += torch.pow(s[num-1-i], 2)
        return result


# class Loss(nn.Module):
#         def __init__(self,input_set):
#             self.input_set = input_set
#         def loss_y(self, w):




# if __name__ == '__main__':
#     source = torch.rand(6, 1000) * -1 # 可以理解为源域有64个14维数据
    # target = torch.rand(4, 5)  # 可以理解为源域有32个14维数据
    # MMD = MMDLoss()
    # print(source)
    # a = MMD(source=source, target=target)

    # x1 = torch.ones(0, 14)
    # x2 = torch.randn(1, 14)
    # x3 = torch.cat((x1, x2), 0)
    # x4 = torch.cat((x3, x2), 0)
    # print(x4)
#     adv = AdversarialLoss()
#     y = adv(source, target)
#     print(y)
    # print(source)

    # BSS = BatchSpectralShrinkage()
    # b = BSS(source)
    # print(b)




# class LambdaSheduler(nn.Module):
#     def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
#         super(LambdaSheduler, self).__init__()
#         self.gamma = gamma
#         self.max_iter = max_iter
#         self.curr_iter = 0

#     def lamb(self):
#         p = self.curr_iter / self.max_iter
#         lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
#         return lamb
    
#     def step(self):
#         self.curr_iter = min(self.curr_iter + 1, self.max_iter)


# class AdversarialLoss(nn.Module):
#     '''
#     Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
#     '''
#     def __init__(self, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, **kwargs):
#         super(AdversarialLoss, self).__init__()
#         self.domain_classifier = Discriminator()
#         self.use_lambda_scheduler = use_lambda_scheduler
#         if self.use_lambda_scheduler:
#             self.lambda_scheduler = LambdaSheduler(gamma, max_iter)
        
#     def forward(self, source, target):
#         lamb = 1.0
#         if self.use_lambda_scheduler:
#             lamb = self.lambda_scheduler.lamb()
#             self.lambda_scheduler.step()
#         source_loss = self.get_adversarial_result(source, True, lamb)
#         target_loss = self.get_adversarial_result(target, False, lamb)
#         adv_loss = 0.5 * (source_loss + target_loss)
#         return adv_loss
    
#     def get_adversarial_result(self, x, source=True, lamb=1.0):
#         x = ReverseLayerF.apply(x, lamb)
#         domain_pred = self.domain_classifier(x)
#         device = domain_pred.device
#         if source:
#             domain_label = torch.ones(len(x), 1).long()
#         else:
#             domain_label = torch.zeros(len(x), 1).long()
#         loss_fn = nn.BCELoss()
#         loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
#         return loss_adv

# class ReverseLayerF(Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x.view_as(x)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha
#         return output, None

# class Discriminator(nn.Module):
#     def __init__(self, input_dim=256, hidden_dim=256): 
#         super(Discriminator, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         layers = [
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         ]
#         self.layers = torch.nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.layers(x)





# class Lambda_loss(nn.Module):
#     def __init__(self):
#         self.k = 1
        
#     def lambda_loss(outputs,labels,auc1,train_auc_history):
#         criterion = nn.BCELoss()
#         Lam = []
#         for auc_f in auc1:
#             x = (float(auc_f) - float(train_auc_history[-1])) / 0.05
#             lam = 1. / (1 + np.exp(x))
#             Lam.append(lam)
#         Index = []
#         for k in range(len(outputs.tolist())):
#             nums = np.array(labels.tolist()[k])
#             index = np.where(nums==1.0)[0].tolist()
#             Index.append(index)
#         Loss1 = []
#         for kk in range(len(labels.tolist()[0])):
#             OUT,LAB = [], []
#             for t in range(len(Index)):
#                 for r in Index[t]:
#                     if r == kk :
#                         OUT.append(outputs.tolist()[t])
#                         LAB.append(labels.tolist()[t])
#             if OUT:
#                 OUT1 = torch.tensor(OUT)
#                 LAB1 = torch.tensor(LAB)
#                 loss = criterion(OUT1, LAB1)
#                 print(type(loss))
#             else:
#                 loss = 0.0
#             Loss1.append(loss)
#         loss = 0
#         for k in range(len(Loss1)):
#             l1 = lam[k] * Loss1[k]
#             loss += l1
        # return tensor(loss).requires_grad_(True) 
