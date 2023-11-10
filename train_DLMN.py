import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import random
from dataset import CityscapesDatasetSSL, CityscapesDataset
import torch
import numpy as np
import torch.optim as optim
# import os
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from utils import *
import torch.autograd as autograd
from deeplab import Res_Deeplab
from fcn8s import VGG16_FCN8s
from torchvision import transforms
from metrics import StreamSegMetrics
from lovasz_losses import lovasz_softmax

import kornia

########################################<---Head--->
import warnings

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class BlockMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask, input_mask
########################################<---Tail--->

def color_jitter(color_jitter, data=None, s=0.2, p=0.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                data = seq(data)
    return data


def gaussian_blur(blur, data=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data

def train(FLAGS):
    device = torch.device('cpu') # set the device to cpu
    if(torch.cuda.is_available()): # check if cuda is available
        device=torch.randn([1,3,1,4]).cuda().device
    torch.cuda.empty_cache()
    writer = SummaryWriter('./runs/' + FLAGS.runs)

    ###############################################################
    ############ loading training + validation data ###############
    ###############################################################

    city_val_dst = CityscapesDataset(
        label_root='./datasets/cityscape/gtFine',
        rgb_root='./datasets/cityscape/leftImg8bit',
        label_path='./datasets/cityscapes_val.txt',
        rgb_path='./datasets/rgb_cityscapes_val.txt',
        crop_size=(1024, 512),
    )

    dataset = CityscapesDatasetSSL(
        root='./datasets/cityscape',
        list_path='./datasets/rgb_cityscapes_train.txt',
        crop_size=(1024, 512),
        label_folder=None,
        mirror=True
    )

    print('using ', len(dataset), ' images for training')

    class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]).to(device=device)

    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=FLAGS.num_channels, reduction='none')

    scaler = GradScaler()

    ###############################################################
    ###################### loading model ##########################
    ###############################################################

    if FLAGS.model == 'deeplab':
        gta_model = Res_Deeplab(num_classes=19).to(device)
        last_model = Res_Deeplab(num_classes=19).to(device)
    elif FLAGS.model == 'fcn8s':
        gta_model = VGG16_FCN8s(num_classes=19).to(device)

    if FLAGS.load_saved:

        ### In practice, we use GtA adapted model as initialization ###

        # last_model.load_state_dict(torch.load('./checkpoints/dl_synthia_pl_allg_round3_best.pth'))#_reproduced.pth from dl_synthia_allg_except_gd.pth
        last_model.load_state_dict(torch.load('./checkpoints/dl_gta5_adapted_wo_cpae.pth'))#_official.pth
        last_model.eval()#
        # gta_model.load_state_dict(torch.load('./checkpoints/dl_synthia_pl_allg_round3_best.pth'))#_reproduced.pth from dl_synthia_allg_except_gd.pth
        gta_model.load_state_dict(torch.load('./checkpoints/dl_gta5_adapted_wo_cpae.pth'))#_official.pth

    gta_model.train()
    print('gta model loading done')

    if FLAGS.model == 'deeplab':
        for name, param in gta_model.named_parameters():
            if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False
        pa = [
            {'params': gta_model.layer1.parameters(), 'lr': FLAGS.lr * 1.0},
            {'params': gta_model.layer2.parameters(), 'lr': FLAGS.lr * 1.0},
            {'params': gta_model.layer3.parameters(), 'lr': FLAGS.lr * 1.0},
            {'params': gta_model.layer4.parameters(), 'lr': FLAGS.lr * 1.0},
            ]
        optimizer = optim.SGD(pa, lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iteration: (1 - iteration / FLAGS.end_iter) ** 0.9)
    elif FLAGS.model == 'fcn8s':
        for name, param in gta_model.named_parameters():
            if 'conv2' not in name and 'conv3' not in name and 'conv4' not in name:
                param.requires_grad = False

        optimizer = optim.Adam(
            [
                {'params': gta_model.get_parameters(bias=False)},
                {'params': gta_model.get_parameters(bias=True),
                 'lr': FLAGS.lr * 2}
            ],
            lr=FLAGS.lr,
            betas=(0.9, 0.99))

    cityvalloader = data.DataLoader(city_val_dst, batch_size=10, shuffle=False, num_workers=4)
    citytrainloader = data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6, drop_last=True)

    city_iter = iter(citytrainloader)

    if not os.path.exists('./checkpoints'):
        os.makedirs('checkpoints')

    val_metrics = StreamSegMetrics(19)
    train_metrics = StreamSegMetrics(19)

    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (1,3,1,1))
    mean_img = torch.zeros(1, 1)

    def validate(iteration, loader):
        val_metrics.reset()
        gta_model.eval()
        with torch.no_grad():
            for i, (batch, rgb_batch) in enumerate(loader):
                rgb_batch = rgb_batch.to(device=device, dtype=torch.float)
                batch = batch.to(device=device, dtype=torch.int64)

                input_size = rgb_batch.size()[2:]
                pred = gta_model(rgb_batch)
                pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
                preds = pred.detach().max(dim=1)[1].cpu().numpy()
                targets = batch.cpu().numpy()
                val_metrics.update(targets, preds)

                if i % 200 == 0:
                    grd = segMap3(rgb_batch, batch, pred)
                    writer.add_image('city/rgb_label_pred', grd, i + iteration)

            score = val_metrics.get_results()
        return score

    add_img_th = FLAGS.save_every # after _ iterations, add images

    print('started training')
    if FLAGS.entw != 0.0:
        print('Entropy will be used for training')
    else:
        print('No entropy minimization')

    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False

    for iteration in range(FLAGS.start_iter, FLAGS.end_iter):
        gta_model.train()

        if FLAGS.model == 'fcn8s':
            gta_model.adjust_learning_rate(FLAGS, optimizer, iteration)

        optimizer.zero_grad()

        train_metrics.reset()
        index = iteration

        try:
            psu_lbl, tgt_img = city_iter.next()
        except StopIteration:
            print('StopIteration occurred at ', iteration)
            city_iter = iter(citytrainloader)
            psu_lbl, tgt_img = city_iter.next()
        ###
        tgt_img_aug = tgt_img / 255.0
        tgt_img_aug = color_jitter(color_jitter=random.uniform(0, 1), data=tgt_img_aug, s=0.2, p=0.2)
        tgt_img_aug = gaussian_blur(blur=random.uniform(0, 1), data=tgt_img_aug)
        tgt_img_aug = tgt_img_aug * 255.0
        ###
        ###
        mask_gen = BlockMaskGenerator(mask_ratio=0.5, mask_block_size=64)
        tgt_img_aug_masked, input_mask = mask_gen.mask_image(tgt_img_aug)
        ###
        if mean_img.shape[-1] < 2:
            B, C, H, W = tgt_img.shape
            mean_img = IMG_MEAN.repeat(B,1,H,W).to(device)

        # tgt_img, psu_lbl = Variable(tgt_img).cuda(), Variable(psu_lbl.long()).cuda()
        tgt_img = Variable(tgt_img).cuda()
        tgt_img_aug = Variable(tgt_img_aug).cuda()
        tgt_img_aug_masked = Variable(tgt_img_aug_masked).cuda()

        # RGB to BGR (since not doing RGB to BGR inside dataloader)
        tgt_img = torch.flip(tgt_img, [1])
        tgt_img_aug = torch.flip(tgt_img_aug, [1])
        tgt_img_aug_masked = torch.flip(tgt_img_aug_masked, [1])

        tgt_img = tgt_img.clone() - mean_img
        tgt_img = tgt_img.detach()
        tgt_img_aug = tgt_img_aug.clone() - mean_img
        tgt_img_aug = tgt_img_aug.detach()
        tgt_img_aug_masked = tgt_img_aug_masked.clone() - mean_img
        ###
        # tgt_img_aug_masked = tgt_img_aug * Variable(input_mask).cuda()# masking (multiply by zero) after normalization
        ###
        tgt_img_aug_masked = tgt_img_aug_masked.detach()

        input_size = tgt_img.size()[2:]

        with torch.no_grad():# generate pseudo-labels
            ###
            x = last_model.conv1(tgt_img)
            x = last_model.bn1(x)
            x = last_model.relu(x)
            x = last_model.maxpool(x)
            x = last_model.layer1(x)
            x = last_model.layer2(x)
            x = last_model.layer3(x)
            x = last_model.layer4(x)
            feas = x
            listL = []
            Pavg = None
            num = 10
            for i in range(num):
                Pi = last_model.layer5(nn.Dropout2d(p=0.2)(feas))
                Pi = F.interpolate(Pi, size=input_size, mode='bilinear', align_corners=True)
                listL.append(Pi)
                if i==0:
                    Pavg = F.softmax(Pi, dim=1)
                else:
                    Pavg = Pavg + F.softmax(Pi, dim=1)
            
            #PLG
            Pavg/=num
            pl = Pavg.argmax(dim=1)
            psu_lbl = Variable(pl.long()).cuda()# early teacher

            #PLS
            Pavg = Pavg * torch.log(Pavg)
            Pavg = -1.0 * Pavg.sum(dim=1)
            for i in range(num):
                if i==0:
                    Pbhd = F.softmax(listL[i], dim=1) * F.log_softmax(listL[i], dim=1)
                else:
                    Pbhd = Pbhd + F.softmax(listL[i], dim=1) * F.log_softmax(listL[i], dim=1)
            Pbhd/=num
            Pbhd = Pbhd.sum(dim=1)
            
            #coefficient
            co = -1.0 * torch.log(Pavg + Pbhd + 1e-5)
            ###

            ###
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            augPavg = last_model(tgt_img)#regularization
            augPavg = F.interpolate(augPavg, size=input_size, mode='bilinear', align_corners=True)
            for s in scales:
                tgt_img_s = F.interpolate(tgt_img, scale_factor=s, mode='bilinear', align_corners=True)
                augPavg_s = gta_model(tgt_img_s)
                augPavg += F.interpolate(augPavg_s, size=input_size, mode='bilinear', align_corners=True)

            augPavg /= (len(scales)+1)
            augPavg = augPavg.argmax(dim=1)
            augPavg = Variable(augPavg.long()).cuda()# late teacher
            ###

        with autocast():# compute L_{baseline}
            pred = gta_model(tgt_img_aug)
            loss = 0
            if FLAGS.entw != 0.0:
                # Entropy computation taken from FDA (CVPR2020) code
                P = F.softmax(pred, dim=1)
                logP = F.log_softmax(pred, dim=1)
                PlogP = P * logP
                ent = -1.0 * PlogP.sum(dim=1)
                # max. entropy for 19 classes is 2.9444 so normalizing
                ent = ent / 2.9444
                # computing robust entropy (penalty function from FDA paper)
                ent = ent ** 2.0  + 1e-8
                ent = ent ** 2.0
                loss_ent = ent.mean()
            pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
            
            if FLAGS.loss == 'ce':
                loss_seg = ce_loss(pred, psu_lbl)
                loss_seg = loss_seg * co
                loss_seg = loss_seg.mean()
            elif FLAGS.loss == 'lovasz':
                output_sm = F.softmax(pred, dim=1)
                loss_seg = lovasz_softmax(output_sm, psu_lbl, ignore=19)
            
            loss = loss_seg
            if FLAGS.entw != 0.0:
                loss += FLAGS.entw * loss_ent

            ###L_{whole}
            output_sm = F.softmax(pred, dim=1)
            loss += lovasz_softmax(output_sm, augPavg, ignore=19)
            ###
        scaler.scale(loss).backward()

        ####################################<---Head--->
        with autocast():#L_{ccl}
            x_dt = gta_model.conv1(tgt_img_aug)
            x_dt = gta_model.bn1(x_dt)
            x_dt = gta_model.relu(x_dt)
            x_dt = gta_model.maxpool(x_dt)
            x_dt = gta_model.layer1(x_dt)
            x_dt = gta_model.layer2(x_dt)
            x_dt = gta_model.layer3(x_dt)
            x_dt = gta_model.layer4(x_dt)
            feas_dt = x_dt
            pred = gta_model.layer5(nn.Dropout2d(p=0.95)(feas_dt))
            loss = 0
            pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
            output_sm = F.softmax(pred, dim=1)
            loss_seg = lovasz_softmax(output_sm, augPavg, ignore=19)
            loss = loss_seg * 1.0
        scaler.scale(loss).backward()
        
        with autocast():#L_{pcl}
            pred = gta_model(tgt_img_aug_masked)
            loss = 0
            pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
            output_sm = F.softmax(pred, dim=1)
            loss_seg = lovasz_softmax(output_sm, augPavg, ignore=19)
            loss = loss_seg * 1.0
        scaler.scale(loss).backward()
        ####################################<---Tail--->
        
        scaler.step(optimizer)
        scaler.update()
        if FLAGS.model == 'deeplab':
            scheduler.step()

        # writer.add_scalar('loss/ce_loss', loss.item(), index)

        if index==0:
            city_miou = validate(index, cityvalloader)
            print(city_miou)
            print('<------------------>')

        # adding image
        if (index + 1) % add_img_th == 0:
            grid = segMap3(tgt_img, psu_lbl, pred)
            writer.add_image('training/rgb_label_pred', grid, index)
            city_miou = validate(index, cityvalloader)
            if FLAGS.source == 'gta5':
                miou_option = 'Mean IoU'
                print('19 classes ................ miou ', city_miou[miou_option])
            elif FLAGS.source == 'synthia':
                miou_option = 'Sixteen IoU'
                print('16 classes ................ miou ', city_miou[miou_option])
            writer.add_scalar('metrics/city_miou', city_miou[miou_option], index)
            if index == (add_img_th - 1):
                max_miou=city_miou[miou_option]
                torch.save(gta_model.state_dict(), os.path.join('./checkpoints', FLAGS.runs + '_' + 'best' + '.pth'))
                print('done saving')
            else:
                if city_miou[miou_option]>max_miou:
                    max_miou=city_miou[miou_option]
                    torch.save(gta_model.state_dict(), os.path.join('./checkpoints', FLAGS.runs + '_' + 'best' + '.pth'))
                    print('done saving')
    # End of training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--num_channels', type=int, default=19, help="number of channels in the image")
    parser.add_argument('--save_every', type=int, default=200, help='saving images after every _ iterations')

    parser.add_argument('--start_iter', type=int, default=0, help='iteration number to start from')
    parser.add_argument('--end_iter', type=int, default=100000, help='iteration number for stopping')

    parser.add_argument('--lr', type=float, default=2.5e-4, help="starting learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum for optimizer")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay param in optimizer")

    parser.add_argument('--model', type=str, choices=['deeplab', 'fcn8s'], default='deeplab')
    parser.add_argument('--source', type=str, choices=['gta5', 'synthia'], default='gta5')
    parser.add_argument('--loss', type=str, choices=['ce', 'lovasz'], default='ce')

    parser.add_argument('--entw', type=float, default=0.0, help="weight for entropy loss")

    parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
    parser.add_argument('--runs', type=str, help='tensorboard runs folder name and model weights save name')

    FLAGS = parser.parse_args()

    SEED = 3407
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train(FLAGS)
