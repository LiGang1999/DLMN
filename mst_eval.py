import time
import argparse
from dataset import CityscapesDataset
import torch
import sys
import os
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils import *
from deeplab import Res_Deeplab
from fcn8s import VGG16_FCN8s
from metrics import StreamSegMetrics

def eval(FLAGS):
    device = torch.device('cpu') # set the device to cpu
    if(torch.cuda.is_available()): # check if cuda is available
        # device = torch.device('cuda:0') # if cuda, set device to cuda
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        device=torch.randn([1,3,1,4]).cuda().device
    torch.cuda.empty_cache()

    if FLAGS.visualize:
        writer_name = FLAGS.dataset + '_' + FLAGS.load_model
        writer = SummaryWriter('runs/' + writer_name)
        print('visualizing at runs/', writer_name)

    if FLAGS.model == 'deeplab':
        gta_model = Res_Deeplab(num_classes=19).to(device)
    elif FLAGS.model == 'fcn8s':
        gta_model = VGG16_FCN8s(num_classes=19).to(device)
    print('loading model from ', FLAGS.load_model)
    gta_model.load_state_dict(torch.load(FLAGS.load_model))
    gta_model.eval()
    print('gta model loading done')

    valid_dst = CityscapesDataset(
        label_root='./datasets/cityscape/gtFine',
        rgb_root='./datasets/cityscape/leftImg8bit',
        label_path='./datasets/cityscapes_val.txt',
        rgb_path='./datasets/rgb_cityscapes_val.txt',
        crop_size=(1024, 512),
    ) 
    
    valloader = data.DataLoader(valid_dst, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    val_metrics = StreamSegMetrics(19)
    scales = [float(_) for _ in str(FLAGS.test_scales).split(',')]
    
    if 1 in scales:
        scales.remove(1)
    
    def validate(loader):
        add_img = 100
        val_metrics.reset()
        print('scales', scales)
        with torch.no_grad():
            for i, (batch, rgb_batch) in enumerate(loader):
                rgb_batch = rgb_batch.to(device=device, dtype=torch.float)
                batch = batch.to(device=device, dtype=torch.int64)
                input_size = rgb_batch.size()[2:]
                
                pred = gta_model(rgb_batch)
                pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
                for s in scales:
                    rgb_batch_s = F.interpolate(rgb_batch, scale_factor=s, mode='bilinear', align_corners=True)
                    pred_s = gta_model(rgb_batch_s)
                    pred += F.interpolate(pred_s, size=input_size, mode='bilinear', align_corners=True)

                pred /= (len(scales)+1)

                preds = pred.detach().max(dim=1)[1].cpu().numpy()
                targets = batch.cpu().numpy()
                val_metrics.update(targets, preds)
                if i % add_img == 0 and FLAGS.visualize:
                    grd = segMap3(rgb_batch, batch, pred)
                    writer.add_image('eval/rgb_label_pred', grd, i)
                elif i % add_img == 0:
                    print(i, val_metrics.get_results()['Mean IoU'])

            score = val_metrics.get_results()
        return score

    score = validate(valloader)
    print('Mean IoU: ',score['Mean IoU'])
    print('Sixteen IoU: ',score['Sixteen IoU'])
    print('Thirteen IoU: ',score['Thirteen IoU'])
    print('Class IoU: ',score['Class IoU'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['deeplab', 'fcn8s'], default='deeplab')
    parser.add_argument('--load_model', type=str, default='checkpoints/model2_gta', help='path to model weights to be evaluated')
    parser.add_argument('--visualize', type=bool, default=False, help='whether to visualize eval')
    parser.add_argument('--test_scales', type=str, default='1')
    FLAGS = parser.parse_args()
    eval(FLAGS)
