import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.vgg import vgg16
from ssim import ssim, ms_ssim, SSIM, MS_SSIM
from SSM_LIC_arch import SSM_LIC
from wavelet import DWT,IWT
from SCARM import SCARM


def data_transform(X):
    return 2 * X - 1.0
def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ssm_enhance = SSM_LIC()
        self.scarm = SCARM(in_channels=3, out_channels=64)



    def forward(self, Input, target,training=True):

        data_dict = {}
        dwt, idwt = DWT(), IWT()
        n, c, h, w = Input.shape

        input_img_norm = data_transform(Input)
        input_dwt = dwt(input_img_norm)
        input_LL0, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        if training:
            gt_img_norm = data_transform(target)
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            x_freq = torch.fft.rfft2(gt_high0, norm='backward')
            # print(x_freq.shape)
            tar_mag = torch.abs(x_freq)
            tar_pha = torch.angle(x_freq)

            input_LL = self.ssm_enhance(input_LL0)
            input_high0, x_mag, x_pha = self.scarm(input_high0)

            pred_LL = idwt(torch.cat((input_LL, input_high0), dim=0))

            pred_x = inverse_data_transform(pred_LL)

            data_dict["input_high0"] = input_high0
            data_dict["gt_high0"] = gt_high0

            data_dict["pred_LL"] = input_LL
            data_dict["gt_LL"] = gt_LL

            data_dict["x_mag"] = x_mag
            data_dict["x_pha"] = x_pha
            data_dict["tar_mag"] = tar_mag
            data_dict["tar_pha"] = tar_pha

            data_dict["pred_x"] = pred_x

        else:
            input_LL = self.ssm_enhance(input_LL0)
            input_high0= self.scarm(input_high0)

            pred_LL = idwt(torch.cat((input_LL, input_high0), dim=0))

            pred_x = inverse_data_transform(pred_LL)
            data_dict["pred_x"] = pred_x
        return data_dict

class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class MESA(nn.Module):
    def __init__(self, opt):
        super(MESA, self).__init__()
        self.device = torch.device(opt.device)
        self.decoder = Net().to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)
        self.ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True).to(self.device)
        self.VGG16 = PerceptionLoss().to(self.device)
        self.l2_loss = torch.nn.MSELoss()

        self.TV_loss = TVLoss()

    def forward(self, Input, target, training=True):
        self.Input = Input
        self.target = target

        # self.label = label
        if training:
            self.out = self.decoder.forward(Input,target)

    def val(self, Input,target,val=False):
        if val:
            self.out = self.decoder.forward(Input, target,training=False)
            return self.out

    def sample(self, testing=False):
        if testing:
            self.out = self.decoder.forward(self.Input,self.target,training=False)
            return self.out

    def elbo(self, opt,Input,target, analytic_kl=True):
        input_high0, gt_high0, = self.out["input_high0"], \
                                 self.out["gt_high0"],

        pred_LL, gt_LL, pred_x,  = self.out["pred_LL"], self.out["gt_LL"], self.out["pred_x"]


        x_mag, x_pha, tar_mag, tar_pha= self.out["x_mag"], self.out["x_pha"], self.out["tar_mag"], self.out["tar_pha"]
        self.Ldft = 1.0 *(self.L1_loss(x_mag, tar_mag)) + 1.0 *(self.L1_loss(x_pha, tar_pha))


        self.frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
                                # self.l2_loss(input_high1, gt_high1) +
                                self.l2_loss(pred_LL, gt_LL)) + \
                         0.01 * (self.TV_loss(input_high0) +
                                # self.TV_loss(input_high1) +
                                self.TV_loss(pred_LL))


        mseloss = self.L1_loss(pred_x, target)
        _ssim_loss = 1 - self.ssim_loss(target, pred_x)

        self.g_loss = 0.2 * mseloss + _ssim_loss * 0.8

        self.reconstruction_loss = self.criterion(pred_x, target)
        self.vgg16_loss = self.VGG16(pred_x, target)
        return  self.vgg16_loss + self.g_loss + self.frequency_loss + self.Ldft

