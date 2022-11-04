import datetime
from pathlib import Path
from argparse import ArgumentParser
from typing import List
from self_supervised import MriSelfSupervised
from ssl_transform import SslTransform
import pytorch_lightning as pl

from fastmri.pl_modules import FastMriDataModule

from fastmri import fft2c,ifft2c
from fastmri.data import subsample
import numpy as np
import torch.optim
from torch.utils.data import DataLoader

#new import 
from torchsummary  import summary
from ploting import imsshow
import random
from torch.utils.tensorboard import SummaryWriter
random.seed(42)

writer = SummaryWriter("/home/liuchun/ssdu_git/fast_MRI/fastMRI/self_supervised/runs/test") 
def handle_args():
    parser = ArgumentParser()

    #num_gpus = 0
    #backend = "ddp_cpu"
    #batch_size = 1 if backend == "ddp" else num_gpus
    #batch_size = 2

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        dest='mode',
        help="Operation mode"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        type=str,
        dest='device',
        help="Device type",
    )
    parser.add_argument(
        "--data_path",
        default='/data/fastmri', #设置数据路径
        type=Path,
        # required=True,
        dest='data_path',
        help="Path to data",
    )
    parser.add_argument(
        "--checkpoint_path",
        default='/home/liuchun/ssdu_git/fast_MRI/fastMRI/self_supervised/save_model/checkpoint1', #设置存储模型路径
        type=Path,
        # required=True,
        dest='checkpoint_path',
        help="When train, dir path for saving model checkpoints; when test, either director (from which to load newest"
             " checkpoint) or specific checkpoint file to load",
    )
    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",  
    )
    parser.add_argument(
        "--accelerator",
        dest='accelerator',
        default='ddp',
        help="What distributed version to use",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument("--non_deterministic", action='store_false', default=True, dest='deterministic')
    parser.add_argument("--replace_sampler_ddp", action='store_true', default=False, dest='replace_sampler_ddp',
                        help="Replace sampler ddp")
    parser.add_argument("--seed", default=42, dest='seed', help="Seed for all the random generators")
    parser.add_argument("--num_gpus", default=1, help="The number of available GPUs (when device is 'cuda'")

    return parser.parse_args()


def get_sorted_checkpoint_files(checkpoint_dir: Path) -> List[Path]:
    files = list(checkpoint_dir.glob('*.pt'))
    files.sort()
    return files


def save_checkpoint(model: torch.nn.Module, checkpoint_dir: Path, limit=None):
    filename = 'ssl_sd_checkpoint_{}.pt'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
        torch.save(model.state_dict(), checkpoint_dir.joinpath(filename))
    else:
        torch.save(model.state_dict(), checkpoint_dir.joinpath(filename))
        files = get_sorted_checkpoint_files(checkpoint_dir)
        if limit and len(files) > limit:
            files[0].unlink()


def load_from_checkpoint(model: torch.nn.Module, checkpoint_dir: Path, specific_file: str = None, set_eval=True):
    if specific_file is None:
        files = get_sorted_checkpoint_files(checkpoint_dir)
        file_path = files[-1]
    else:
        file_path = checkpoint_dir.joinpath(specific_file)
    model.load_state_dict(torch.load(file_path))
    if set_eval:
        model.eval()


def calc_ssl_loss(u, v):
    abs_u_minus_v = torch.abs(u - v)
    #draw picture
    # u1=u[0].unsqueeze(0)
    # v1=v[0].unsqueeze(0)
    # img_show=torch.cat((u1,v1),0)    
    # imsshow(img_show.data.cpu().numpy(),['input','output'],1,cmap='gray',is_colorbar=True,filename2save=f'/home/liuchun/ssdu_git/fast_MRI/fastMRI/self_supervised/images/input/{0}.png')
    abs_u = torch.abs(u)
    term_1 = torch.pow(abs_u_minus_v, 2) / (torch.pow(abs_u, 2) + 10e-8)
    term_2 = abs_u_minus_v / (abs_u+10e-8)
    return torch.mean(term_1 + term_2)

#实现mask 是否是在k空间？ 数据存储形式
#需要传入的数据格式     
#这部分有问题
# def choose_loss_split(volume, ratio=0.5):
#     # TODO: come back and implement overlap
#     # arange = np.arange(volume.shape[0])# 1,320,320
#     arange = np.arange(volume.shape[-1])
#     # theta_indices = np.random.Generator.choice(arange, size=int(volume.shape[0] * ratio), replace=False)
#     # theta_indices = np.random.Generator.choice(arange, size=int(volume.shape[-1] * ratio), replace=False)
#     #从数据中取线 实现采样的目的
#     rng = np.random.default_rng()
#     theta_indices = rng.choice(arange, size=int(320 * ratio), replace=False)

#     lambda_indices = arange[np.isin(arange, theta_indices, invert=True)]
#     volume_theta_view = volume[theta_indices]
#     volume_lambda_view = volume[lambda_indices]
#     return volume_theta_view, volume_lambda_view

####TODO:
def choose_loss_split(volume, ratio=0.5):
    mask_ratio_prob = torch.rand(volume.shape[1], volume.shape[2]) 
    mask_ratio_prob = mask_ratio_prob.unsqueeze(-1).repeat(1,1,2)
    mask_ratio_theta = mask_ratio_prob <= ratio
    mask_ratio_lambda = mask_ratio_prob > ratio
    import matplotlib.pyplot as plt
    # plt.imshow(ifft2c(volume).cpu().detach().numpy()[0, :, :, 0], cmap='bone')
    # plt.title('before samplinng')
    # plt.show()
    volume_theta_view = mask_ratio_theta * volume
    volume_lambda_view = mask_ratio_lambda * volume
    # plt.imshow(ifft2c(volume_theta_view).cpu().detach().numpy()[0, :, :, 0], cmap='bone')
    # plt.title('bafftasdaee samplinng')
    # plt.show()
    return volume_theta_view, volume_lambda_view, mask_ratio_lambda

# def choose_loss_split(volume, ratio=0.5):
#     # TODO: come back and implement overlap
#     # arange = np.arange(volume.shape[0])# 1,320,320
#     un_kspace=fftc(volume)
#     arange = np.arange(volume.shape[-1])
#     # theta_indices = np.random.Generator.choice(arange, size=int(volume.shape[0] * ratio), replace=False)
#     # theta_indices = np.random.Generator.choice(arange, size=int(volume.shape[-1] * ratio), replace=False)
#     #从数据中取线 实现采样的目的
#     rng = np.random.default_rng()
#     theta_indices = rng.choice(arange, size=int(320 * ratio), replace=False)

#     lambda_indices = arange[np.isin(arange, theta_indices, invert=True)]
#     volume_theta_view = volume[theta_indices]
#     volume_lambda_view = volume[lambda_indices]
#     return volume_theta_view, volume_lambda_view




def run_training_for_volume(volume, model: torch.nn.Module, optimizer,device):
    #  masked_kspace, image, target, mean, std, fname, slice_num, max_value
    volume_theta_view, volume_lambda_view, mask_lambda = choose_loss_split(volume[0])# input 320 320
    volume_theta_view_image = ifft2c(volume_theta_view)
    mask_lambda = mask_lambda.to(device)
    volume_theta_view_image, volume_lambda_view=volume_theta_view_image.to(device), volume_lambda_view.to(device) #new to model
    volume_theta_view_image.requires_grad=True
    volume_lambda_view.requires_grad=True
    volume_theta_view_image = volume_theta_view_image.permute(0,3,1,2)
    prediction, _, _, _ = model(volume_theta_view_image, None, torch.ones_like(mask_lambda.float()) - mask_lambda.float(), volume_theta_view_image) 
    #应该在k空间做损失？？  上面的逻辑不正确   输出数据转换到k空间 加上lambda的k空间数据 在采样 数据划分部分 应该用k空间采样 而不是直接在图像域
    loss = calc_ssl_loss(u=volume_lambda_view, v=fft2c(prediction.permute(0,2,3,1))*mask_lambda)
    optimizer.zero_grad()
    loss.backward()  #为了得到梯度
    optimizer.step()
    return loss

def run_training(model: torch.nn.Module, checkpoint_dir: Path, dataloader: DataLoader,device, epochs=100 ):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(epochs):
        # print('len(dataloader):',len(dataloader))
        num =0 
        for volume in dataloader:
            loss = run_training_for_volume(volume, model, optimizer,device)
            num += 1
            print(loss)
        # with torch.no_grad():
        #     import matplotlib.pyplot as plt
        #     volume_theta_view, volume_lambda_view, mask_lambda = choose_loss_split(volume[0])# input 320 320
        #     volume_theta_view_image = ifft2c(volume_theta_view)
        #     mask_lambda = mask_lambda.to(device)
        #     volume_theta_view_image, volume_lambda_view=volume_theta_view_image.to(device), volume_lambda_view.to(device) #new to model
        #     volume_theta_view_image = volume_theta_view_image.permute(0,3,1,2)
        #     prediction, _, _, _ = model(volume_theta_view_image, None, torch.ones_like(mask_lambda.float()) - mask_lambda.float(), volume_theta_view_image) 
        #     plt.imshow(prediction[0, 0, :, :].cpu().detach().numpy())
        #     plt.show()
        # #实现保留loss最小的损失函数
        # writer.add_scalar("loss(train)",e, epoch_loss / num) #实现tensorboard保存内通过
        save_checkpoint(model, checkpoint_dir)
        print(f"loss: {loss:>7f}  [{e:>5d}/{epochs:>5d}]")


def run_pretrained_inference(model, checkpoint_source: Path, dataloader, device):
    # TODO: source may be directory (newest file) or actual file to load
    # TODO: implement
    model = MriSelfSupervised()
    model.to(device)
    state_dict = torch.load(checkpoint_source)
    model.load_state_dict(state_dict)
    model.eval()
    num = 0
    import matplotlib.pyplot as plt
    for volume in dataloader:
        num +=1
        volume_theta_view, volume_lambda_view, mask_lambda = choose_loss_split(volume[0])# input 320 320
        volume_theta_view_image = ifft2c(volume_theta_view)
        mask_lambda = mask_lambda.to(device)
        volume_theta_view_image, volume_lambda_view=volume_theta_view_image.to(device), volume_lambda_view.to(device) #new to model
        volume_theta_view_image = volume_theta_view_image.permute(0,3,1,2)
        prediction, _, _, _ = model(volume_theta_view_image, None, torch.ones_like(mask_lambda.float()) - mask_lambda.float(), volume_theta_view_image) 
        loss = calc_ssl_loss(u=volume_lambda_view, v=fft2c(prediction.permute(0,2,3,1))*mask_lambda)
        img = prediction[0, 0, :, :].cpu().detach().numpy()
        print(torch.mean(loss), volume_lambda_view.shape)
        plt.imshow(img, cmap='bone')
        plt.show()
        plt.savefig('/home/liuchun/ssdu_git/fast_MRI/fastMRI/self_supervised/images/outputs' + str(num).zfill(4) + '.png')

def main():
    args = handle_args()
    pl.seed_everything(args.seed)

    # creates k-space mask for transforming
    mask = subsample.create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    train_transform = SslTransform(mask_func=mask, use_seed=False)
    val_transform = SslTransform(mask_func=mask)
    test_transform = SslTransform()

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge='singlecoil',
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split='test',
        # TODO: this in particular might need to be changed
        test_path=None,
        sample_rate=None,
        batch_size=1,
        num_workers=0,
        # distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        distributed_sampler=False,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    checkpoint_path = args.checkpoint_path
    if args.mode == "train":
        if checkpoint_path.exists() and not checkpoint_path.is_dir():
            raise RuntimeError("Existing, non-directory path {} given for checkpoint directory".format(checkpoint_path))
        from torchsummary  import summary
        # dataloader=data_module.train_dataloader()#just want to test
        # print('have a test')
        model = MriSelfSupervised()
        model.to(device)
        run_training(model=model, checkpoint_dir=checkpoint_path, dataloader=data_module.train_dataloader(),device=device)
    elif args.mode == "test":
        if not checkpoint_path.exists():
            raise RuntimeError("Non-existing checkpoint file/directory path {}".format(checkpoint_path))
        model = MriSelfSupervised()
        model.to(device)
        run_pretrained_inference(model=model, 
        checkpoint_source=checkpoint_path, 
        dataloader=data_module.val_dataloader(),
        device=device)
    else:
        raise RuntimeError("Unsupported mode '{}'".format(args.mode))


if __name__ == "__main__":
    main()
