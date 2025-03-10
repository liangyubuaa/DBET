#coding: utf-8
import cv2
import mmcv
import numpy as np
import os
import torch
 
from mmdet.apis import inference_detector, init_detector
 
def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    return heatmap
 
def draw_feature_map(model, img_path, save_dir):
    '''

    '''
    img = mmcv.imread(img_path)
    modeltype = str(type(model)).split('.')[-1].split('\'')[0]
    model.eval()
    model.draw_heatmap = True
    featuremaps = inference_detector(model, img)  
    i=2
    for featuremap in featuremaps:
        heatmap = featuremap_2_heatmap(featuremap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  
        heatmap = np.uint8(255 * heatmap)  
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
        superimposed_img = heatmap * 0.5 + img*0.3  
        cv2.imwrite(os.path.join(save_dir,'cam_'+'P'+str(i)+'.png'), superimposed_img)  
        i=i+1
 
 
from argparse import ArgumentParser
 
def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('save_dir', help='Dir to save heatmap image')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
 
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    draw_feature_map(model,args.img,args.save_dir)
 
if __name__ == '__main__':
    main()