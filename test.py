import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import os
import argparse
import warnings
from PIL import Image
from tqdm import tqdm

import model
import network
import PWCNet

# Suppress autograd warning from correlation package and 
warnings.filterwarnings("ignore", message="Legacy autograd function with non-static forward method is deprecated and will be removed in 1.3.")

if __name__ == '__main__':

    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", type=str, required=True, help='path to scene folder containing high-low resolution frames.')
    parser.add_argument("--interpolation_sequence", type=int, nargs='+', required=True, help='number of frames to be interpolated. To go from 30 fps to 400 fps (not divisible) sequence: [13, 12, 12]')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    # Initialize CNNs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    flowComputation = PWCNet.pwc_dc_net('PWCNet/pwc_net.pth.tar').eval()
    flowComputation.to(device)
    for param in flowComputation.parameters():
        param.requires_grad = False

    flowEnhancement = model.UNet(19, 5)
    flowEnhancement.to(device)
    for param in flowEnhancement.parameters():
        param.requires_grad = False

    appearanceEstimation = model.UNet(201, 3)
    appearanceEstimation.to(device)
    for param in appearanceEstimation.parameters():
        param.requires_grad = False

    resnet = torchvision.models.resnet18(pretrained=True)
    resnetConv1 = nn.Sequential(list(resnet.children())[0])
    resnetConv1[0].stride = (1, 1)
    resnetConv1.to(device)
    for param in resnetConv1.parameters():
            param.requires_grad = False

    # Initialize transforms
    toPILImage = transforms.Compose([transforms.ToPILImage()])
    toTensor   = transforms.Compose([transforms.ToTensor()])

    # Load checkpoints
    dict1 = torch.load('Checkpoints/flowEnhancement.ckpt')
    flowEnhancement.load_state_dict(dict1['state_dict'])
    dict1 = torch.load('Checkpoints/appearanceEstimation.ckpt')
    appearanceEstimation.load_state_dict(dict1['state_dict'])

    # Create output directory
    os.mkdir(os.path.join(args.scene_root, "output"))
    
    # Extract frame paths for low and high resolution frames
    LOW_RES  = "LR"
    HIGH_RES = "HR"

    high_res_names = os.listdir(os.path.join(args.scene_root, HIGH_RES))
    low_res_names  = os.listdir(os.path.join(args.scene_root, LOW_RES))

    frame_counter = 0
    PERIOD = len(args.interpolation_sequence)

    # Get width and height of frames
    im = Image.open(os.path.join(args.scene_root, HIGH_RES, high_res_names[0]))
    width, height = im.size

    # Backwarping function
    testFlowBackWarp = model.backWarp(width, height, device)
    testFlowBackWarp = testFlowBackWarp.to(device)

    with torch.no_grad():
        for high_res_index in tqdm(range(len(high_res_names) - 1)):
            # Open frames
            keyframe_left  = Image.open(os.path.join(args.scene_root, HIGH_RES, high_res_names[high_res_index]))
            keyframe_right = Image.open(os.path.join(args.scene_root, HIGH_RES, high_res_names[high_res_index + 1]))
            keyframe_left  = toTensor(keyframe_left )[None, :, :, :].to(device)
            keyframe_right = toTensor(keyframe_right)[None, :, :, :].to(device)

            low_res_frames = []

            for low_res_index in range(frame_counter, frame_counter + args.interpolation_sequence[high_res_index % PERIOD] + 2):
                low_res_frame = Image.open(os.path.join(args.scene_root, LOW_RES, low_res_names[low_res_index]))
                low_res_frame = toTensor(low_res_frame)[None, :, :, :].to(device)
                low_res_frames.append(low_res_frame)
            
            # Generate intermediate frames
            output_frames = network.testSlomo((flowComputation, flowEnhancement, appearanceEstimation, resnetConv1, testFlowBackWarp),
                                              ((keyframe_left, keyframe_right), low_res_frames),
                                              (height, width))
            
            # Save frames to disk
            [(toPILImage(frame)).save(os.path.join(args.scene_root, "output", str(frame_counter + frame_index + 1) + ".png"))\
                for frame_index, frame in enumerate(output_frames)]
            
            frame_counter += args.interpolation_sequence[high_res_index % PERIOD] + 1