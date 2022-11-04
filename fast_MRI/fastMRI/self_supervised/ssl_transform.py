import fastmri
import numpy as np
import torch
from fastmri.data import transforms as fastmri_transforms
from fastmri.data.subsample import MaskFunc
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt

#new use
from torchvision.utils import save_image

class SslTransform:

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True, is_multicoil: bool = False):
        """
        传入数据 生成mask的函数  随机种子：保证每次生成的随机mask一致  多线圈
        Parameters
        ----------
        mask_func : Optional[MaskFunc]
            A function that can create a mask of appropriate shape. 生成指定尺寸的mask
        use_seed : bool
            If true, this class computes a pseudo random number  伪随机数值
            generator seed from the filename. This ensures that the same
            mask is used for all the slices of a given volume every time.  每次都对slice使用相同的mask
        is_multicoil : bool
            Whether multicoil as opposed to single.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.which_challenge = "multicoil" if is_multicoil else "singlecoil"

    def __call__(self, kspace: np.ndarray, mask: np.ndarray, target: np.ndarray, attrs: Dict, fname: str,
                 slice_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """

        Parameters
        ----------
        kspace
            Input k-space of shape (num_coils, rows, cols) for multi-coil data or (rows, cols) for single coil data.
        mask
            Mask from the test dataset.  几倍降采？
        target
            Target image.
        attrs
            Acquisition related information stored in the HDF5 object.采集相关信息存储在HDF5对象中。
        fname
            File name.
        slice_num
            Serial number of the slice.切片的序列号。

        Returns
        -------
        tuple containing:
                image: Zero-filled input image. 零填充的输入图像
                target: Target image converted to a torch.Tensor. 转换成tensor的图像真值
                mean: Mean value used for normalization.为正则化准备的均值
                std: Standard deviation value used for normalization.为正则化准备的标准差
                fname: File name.图像名
                slice_num: Serial number of the slice. 切片的序列号
        """

        # crop input to correct size 将输入转化成正确图像
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        kspace = fastmri_transforms.to_tensor(kspace)

        image_fully = fastmri.ifft2c(kspace)
        image_fully = fastmri_transforms.complex_center_crop(image_fully, crop_size)
        kspace_cropped = fastmri.fft2c(image_fully)

        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = fastmri_transforms.apply_mask(kspace_cropped, self.mask_func, seed) #得到欠采后的图像
            # print(mask.shape) #show what does mask look  like
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace) #得到原始欠采样图片

        #显示 target内容
        # print(np.unique(target))
        # plt.imshow(target, cmap='bone')
        # plt.show()
        # save_image(torch.from_numpy(np.log(target)),f'/home/liuchun/ssdu_git/fast_MRI/fastMRI/self_supervised/images/lookdata/looktarget.png')




        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = fastmri_transforms.complex_center_crop(image, crop_size)

        
        # absolute value  实现归一化
        image = fastmri.complex_abs(image) #计算得到幅值图像

        # image, mean, std = fastmri_transforms.normalize_instance(image, eps=1e-11)  #只能对幅值图像进行归一化吗

        # kspace=fastmri.fft2c(image)
        # image_abs = fastmri.complex_abs(image) #计算得到幅值图像

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)
            # image_abs = fastmri.rss(image_abs)

        # normalize input 将图像进行归一化
        image, mean, std = fastmri_transforms.normalize_instance(image, eps=1e-11)
        # image, mean, std = fastmri_transforms.normalize_instance(image_abs, eps=1e-11)  #只能对幅值图像进行归一化吗
        image = image.clamp(-6, 6)
        # target = fastmri_transforms.to_tensor(target) #the new line want to confired
        # normalize target 将真值进行归一化
        if target is not None:
            target = fastmri_transforms.to_tensor(target)
            target = fastmri_transforms.center_crop(target, crop_size)
            target = fastmri_transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])
        # return image, target, mean, std, fname, slice_num, max_value,masked_kspace
        masked_kspace = kspace_cropped * mask
        masked_kspace /= torch.max(fastmri.complex_abs(masked_kspace))
        # plt.imshow(fastmri.complex_abs(fastmri.ifft2c(masked_kspace)), cmap='bone')
        # plt.show()
        return masked_kspace, image, target, mean, std, fname, slice_num, max_value
        #max_value 最大值？？
