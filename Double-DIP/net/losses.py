import torch
from torch import nn
import numpy as np
from .layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
from .downsampler import * 
from torch.nn import functional
from scipy.signal import gaussian


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=0, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.use_cuda = True
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
class CannyDiceLoss(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=True):
        super(CannyDiceLoss, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

        self.dice_loss = BinaryDiceLoss()


    def get_canny_edge(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag = grad_mag + torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag = grad_mag + torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation = grad_orientation + 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0
        thresholded[thin_edges>=self.threshold] = 1.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return thresholded
    
    def forward(self, shading, albedo, hybrid):

        canny_shading = self.get_canny_edge(shading)
        canny_albedo = self.get_canny_edge(albedo)
        # canny_hybrid = self.get_canny_edge(hybrid)

        # return self.dice_loss(canny_hybrid, canny_albedo + canny_shading - canny_albedo * canny_shading)
        return (canny_shading * canny_albedo).sum()


class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


class ExtendedL1Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, b, mask):
        normalizer = self.l1(mask, torch.zeros(mask.shape).cuda())
        # if normalizer < 0.1:
        #     normalizer = 0.1
        c = self.l1(mask * a, mask * b) / normalizer
        return c


class NonBlurryLoss(nn.Module):
    def __init__(self):
        """
        Loss on the distance to 0.5
        """
        super(NonBlurryLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x):
        return 1 - self.mse(x, torch.ones_like(x) * 0.5)


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.gray_scale = GrayscaleLayer()
        self.mse = nn.MSELoss().cuda()

    def forward(self, x, y):
        x_g = self.gray_scale(x)
        y_g = self.gray_scale(y)
        return self.mse(x_g, y_g)


class GrayLoss(nn.Module):
    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)


class GradientLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)


class YIQGNGCLoss(nn.Module):
    def __init__(self, shape=5):
        super(YIQGNGCLoss, self).__init__()
        self.shape = shape
        self.var = VarianceLayer(self.shape, channels=1)
        self.covar = CovarianceLayer(self.shape, channels=1)

    def forward(self, x, y):
        if x.shape[1] == 3:
            x_g = rgb_to_yiq(x)[:, :1, :, :]  # take the Y part
            y_g = rgb_to_yiq(y)[:, :1, :, :]  # take the Y part
        else:
            assert x.shape[1] == 1
            x_g = x  # take the Y part
            y_g = y  # take the Y part
        c = torch.mean(self.covar(x_g, y_g) ** 2)
        vv = torch.mean(self.var(x_g) * self.var(y_g))
        return c / vv

class RetinaLoss(nn.Module):
    """
    Gradient loss and its plus version: Exclusion loss
    """
    def __init__(self):

        super(RetinaLoss, self).__init__()
        self.l1_diff = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.AvgPool2d(2)
        self.level = 3
        self.eps = 1e-6
        pass

    @staticmethod
    def compute_gradient(img):
        grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
        grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
        return grad_x, grad_y

    def compute_exclusion_loss(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            if torch.mean(torch.abs(gradx2)) < self.eps or torch.mean(torch.abs(gradx2)) < self.eps:
                gradx_loss.append(0)
                grady_loss.append(0)
                continue

            alphax = 2.0 * torch.mean(torch.abs(gradx1)) / (torch.mean(torch.abs(gradx2)) + self.eps)
            alphay = 2.0 * torch.mean(torch.abs(grady1)) / (torch.mean(torch.abs(grady2)) + self.eps)

            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            gradx_loss.append(torch.mean(torch.mul(torch.pow(gradx1_s, 2), torch.pow(gradx2_s, 2)) ** 0.25))
            grady_loss.append(torch.mean(torch.mul(torch.pow(grady1_s, 2), torch.pow(grady2_s, 2)) ** 0.25))

            img1 = self.downsample(img1)
            img2 = self.downsample(img2)

        loss = 0.5 * (sum(gradx_loss) / float(len(gradx_loss)) + sum(grady_loss) / float(len(grady_loss)))

        return loss

    def compute_gradient_loss(self, img1, img2):
        gradx1, grady1 = self.compute_gradient(img1)
        gradx2, grady2 = self.compute_gradient(img2)

        loss = 0.5 * (self.l1_diff(gradx1, gradx2) + self.l1_diff(grady1, grady2))
        return loss

    def forward(self, img_b, img_r, mode='exclusion'):
        """  Mode in [exclusion/gradient] """
        if mode == 'exclusion':
            loss = self.compute_exclusion_loss(img_b, img_r)
        elif mode == 'gradient':
            loss = self.compute_gradient_loss(img_b, img_r)
        else:
            raise NotImplementedError("mode should in [exclusion/gradient]")
        return loss

