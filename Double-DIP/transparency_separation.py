from net import skip
from net.losses import ExclusionLoss, plot_image_grid, StdLoss, CannyDiceLoss
from net.noise import get_noise
from utils.image_io import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

TwoImagesSeparationResult = namedtuple("TwoImagesSeparationResult",
                                       ["albedo", "shading", "psnr", "alpha1", "alpha2"])

class TwoImagesSeparation(object):
    def __init__(self, image1_name, image2_name, image1, image2, plot_during_training=True, show_every=500, num_iter=4000,
                 original_albedo=None, original_shading=None):
        # we assume the albedo is static
        self.image1 = image1
        self.image2 = image2
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.num_iter = num_iter
        self.loss_function = None
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 2
        self.albedo_net_input = None
        self.shading_net_input = None
        self.original_shading = original_shading
        self.original_albedo = original_albedo
        self.albedo_net = None
        self.shading_net = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.albedo_out = None
        self.shading_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.image1_torch = np_to_torch(self.image1).type(torch.cuda.FloatTensor)
        self.image2_torch = np_to_torch(self.image2).type(torch.cuda.FloatTensor)

    def _init_inputs(self):
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        self.albedo_net_input = get_noise(self.input_depth, input_type,
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.alpha_net1_input = get_noise(self.input_depth, input_type,
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.alpha_net2_input = get_noise(self.input_depth, input_type,
                                          (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.shading_net_input = get_noise(self.input_depth, input_type,
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.albedo_net.parameters()] + \
                          [p for p in self.shading_net.parameters()]
        self.parameters += [p for p in self.alpha1.parameters()]
        self.parameters += [p for p in self.alpha2.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'albedo'
        albedo_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.albedo_net = albedo_net.type(data_type)

        shading_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.shading_net = shading_net.type(data_type)
        alpha_net1 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha1 = alpha_net1.type(data_type)

        alpha_net2 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha2 = alpha_net2.type(data_type)

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self, step):
        reg_noise_std = 0
        albedo_net_input = self.albedo_net_input + (self.albedo_net_input.clone().normal_() * reg_noise_std)
        shading_net_input = self.shading_net_input + \
                                 (self.shading_net_input.clone().normal_() * reg_noise_std)

        self.albedo_out = self.albedo_net(albedo_net_input)
        self.shading_out = self.shading_net(shading_net_input)
        alpha_net_input = self.alpha_net1_input + (self.alpha_net1_input.clone().normal_() * reg_noise_std)
        self.current_alpha1 = self.alpha1(alpha_net_input)[:, :,
                             self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                             self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05

        alpha_net_input = self.alpha_net2_input + (self.alpha_net2_input.clone().normal_() * reg_noise_std)
        self.current_alpha2 = self.alpha2(alpha_net_input)[:, :,
                              self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                              self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1]* 0.9 + 0.05
        self.total_loss = self.mse_loss(self.current_alpha1 * self.albedo_out +
                                        (1 - self.current_alpha1) * self.shading_out,
                                        self.image1_torch)
        self.total_loss += self.mse_loss(self.current_alpha2 * self.albedo_out +
                                         (1 - self.current_alpha2) * self.shading_out,
                                         self.image2_torch)
        self.exclusion = self.exclusion_loss(self.albedo_out, self.shading_out)
        self.total_loss += 0.1 * self.exclusion
        # self.total_loss += self.blur_loss(self.current_alpha2) + self.blur_loss(self.current_alpha1)
        if step < 1000:
            self.total_loss += 0.5 * self.mse_loss(self.current_alpha1,
                                                  torch.tensor([[[[0.5]]]]).type(torch.cuda.FloatTensor))
            self.total_loss += 0.5 * self.mse_loss(self.current_alpha2,
                                                  torch.tensor([[[[0.5]]]]).type(torch.cuda.FloatTensor))

        self.total_loss.backward()

    def _obtain_current_result(self, j):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        albedo_out_np = np.clip(torch_to_np(self.albedo_out), 0, 1)
        shading_out_np = np.clip(torch_to_np(self.shading_out), 0, 1)
        # print(albedo_out_np.shape)
        alpha1 = np.clip(torch_to_np(self.current_alpha1), 0, 1)
        alpha2 = np.clip(torch_to_np(self.current_alpha2), 0, 1)
        v = alpha1 * albedo_out_np + (1 - alpha1) * shading_out_np
        # print(v.shape, self.image2.shape)
        psnr1 = compare_psnr(self.image1, v)
        psnr2 = compare_psnr(self.image2, alpha2 * albedo_out_np + (1 - alpha2) * shading_out_np)
        self.psnrs.append(psnr1+psnr2)
        self.current_result = TwoImagesSeparationResult(albedo=albedo_out_np, shading=shading_out_np,
                                                        psnr=psnr1, alpha1=alpha1, alpha2=alpha2)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f} Exclusion {:5f}  PSRN_gt: {:f}'.format(step,
                                                                                    self.total_loss.item(),
                                                                                    self.exclusion.item(),
                                                                                    self.current_result.psnr),
              '\r', end='')
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            plot_image_grid("albedo_shading_{}".format(step),
                            [self.current_result.albedo, self.current_result.shading])
            # plot_image_grid("learned_mask_{}".format(step),
            #                 [self.current_result.alpha1, self.current_result.alpha2])
            save_image("sum1_{}".format(step), self.current_result.alpha1 * self.current_result.albedo +
                       (1-self.current_result.alpha1)* self.current_result.shading)
            save_image("sum2_{}".format(step), self.current_result.alpha2 * self.current_result.albedo +
                       (1 - self.current_result.alpha2) * self.current_result.shading)

    def finalize(self):
        save_graph(self.image1_name + "_psnr", self.psnrs)
        save_image(self.image1_name + "_albedo", self.best_result.albedo)
        save_image(self.image1_name + "_shading", self.best_result.shading)
        save_image(self.image1_name + "_original", self.image1)
        save_image(self.image2_name + "_original", self.image2)


class Separation(object):
    def __init__(self, image_name, image, plot_during_training=True, show_every=500, num_iter=8000,
                 original_albedo=None, original_shading=None):
        self.image = image
        self.plot_during_training = plot_during_training
        # self.ratio = ratio
        self.psnrs = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.loss_function = None
        # self.ratio_net = None
        self.parameters = None
        self.learning_rate = 0.0005
        self.input_depth = 3
        self.albedo_net_inputs = None
        self.shading_net_inputs = None
        self.original_shading = original_shading
        self.original_albedo = original_albedo
        self.albedo_net = None
        self.shading_net = None
        self.total_loss = None
        self.albedo_out = None
        self.shading_out = None
        self.current_result = None
        self.best_result = None

        self.hybrid_image = None
        self.canny_diced_loss = None

        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.images = create_augmentations(self.image)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]

    def _init_inputs(self):
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        albedo_noise = torch_to_np(get_noise(3,
                                                  input_type,
                                                  (self.images_torch[0].shape[2],
                                                   self.images_torch[0].shape[3])).type(data_type).detach())
        self.albedo_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in create_augmentations(albedo_noise)]
        shading_noise = torch_to_np(get_noise(1,
                                                  input_type,
                                                  (self.images_torch[0].shape[2],
                                                   self.images_torch[0].shape[3])).type(data_type).detach())
        self.shading_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                                      create_augmentations(shading_noise)]

    def _init_parameters(self):
        self.parameters = [p for p in self.albedo_net.parameters()] + \
                          [p for p in self.shading_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'albedo'
        albedo_net = skip(
            3, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.albedo_net = albedo_net.type(data_type)

        shading_net = skip(
            1, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.shading_net = shading_net.type(data_type)

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.canny_diced_loss = CannyDiceLoss(threshold=3.0).type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _get_augmentation(self, iteration):
        if iteration % 2 == 1:
            return 0
        # return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, step):
        if step == self.num_iter - 1:
            reg_noise_std = 0
        elif step < 1000:
            reg_noise_std = (1 / 1000.) * (step // 100)
        else:
            reg_noise_std = 1 / 1000.
        aug = self._get_augmentation(step)
        if step == self.num_iter - 1:
            aug = 0
        albedo_net_input = self.albedo_net_inputs[aug] + (self.albedo_net_inputs[aug].clone().normal_() * reg_noise_std)
        shading_net_input = self.shading_net_inputs[aug] + (self.shading_net_inputs[aug].clone().normal_() * reg_noise_std)


        self.albedo_out = self.albedo_net(albedo_net_input)
        self.shading_out = self.shading_net(shading_net_input)
        # self.hybrid_image = self.albedo_out * self.shading_out

        #self.total_loss = self.l1_loss(self.albedo_out + self.shading_out, self.images_torch[aug])
        self.total_loss = self.l1_loss(self.albedo_out, self.images_torch[aug] / self.shading_out)
        # self.total_loss = self.total_loss + self.canny_diced_loss(self.albedo_out, self.shading_out, self.images_torch[aug])
        # self.total_loss = self.total_loss + 0.01 * self.exclusion_loss(self.albedo_out, self.shading_out)
        self.total_loss.backward()

    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        if step == self.num_iter - 1 or step % 8 == 0:
            albedo_out_np = np.clip(torch_to_np(self.albedo_out), 0, 1)
            shading_out_np = np.clip(torch_to_np(self.shading_out), 0, 1)
            psnr = compare_psnr(self.images[0],  albedo_out_np * shading_out_np)
            self.psnrs.append(psnr)
            self.current_result = SeparationResult(albedo=albedo_out_np, shading=shading_out_np,
                                                   psnr=psnr)
            if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
                self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f}  PSRN_gt: {:f}'.format(step,
                                                                               self.total_loss.item(),
                                                                               self.current_result.psnr),
              '\r', end='')
        if step % self.show_every == self.show_every - 1:
            plot_image_grid("left_right_{}".format(step), [self.current_result.albedo, self.current_result.shading])
            
    def _plot_distance_map(self):
        calculated_left = self.best_result.albedo
        calculated_right = self.best_result.shading
        # this is to left for reason
        # print(distance_to_left.shape)
        pass

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs)
        save_image(self.image_name + "_albedo", self.best_result.albedo)
        save_image(self.image_name + "_shading", self.best_result.shading)
        save_image(self.image_name + "_albedo2", 2 * self.best_result.albedo)
        save_image(self.image_name + "_shading2", 2 * self.best_result.shading)
        save_image(self.image_name + "_original", self.images[0])



SeparationResult = namedtuple("SeparationResult", ['albedo', 'shading', 'psnr'])

if __name__ == "__main__":
    # Separation from two images
    #input1 = prepare_image('images/input1.jpg')
    #input2 = prepare_image('images/input2.jpg')
    #t = TwoImagesSeparation('input1', 'input2', input1, input2, num_iter=7000)
    #t.optimize()
    #t.finalize()

    # Separation of textures
    #t1 = prepare_image('Double-DIP/images/texture12.jpg')
    #t2 = prepare_image('Double-DIP/images/texture16.jpg')

    t1 = prepare_image('reuslts/image/Image0138.png')
    t2 = prepare_image('reuslts/plane_shade.png')
    #s = Separation('textures', (t1+t2)/2)
    t = prepare_image('d_image.png')
    s = Separation('textures', t)
    s.optimize()
    s.finalize()
