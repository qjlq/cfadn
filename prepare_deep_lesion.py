import os
import yaml
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from helper import get_mar_params, simulate_metal_artifact

# Load params
config_file = os.path.join('config', 'dataset.yaml')
splits = ['train', 'test']

with open(config_file) as f:
    config = yaml.safe_load(f)
config = config['deep_lesion']

CTpara = config['CTpara']
for name in CTpara:
    p = CTpara[name]
    if isinstance(p, str):
        CTpara[name] = eval(p)

# Load meta data
npz_file = os.path.join(config['mar_dir'], 'SampleMasks.npz')
data = np.load(npz_file)
metal_masks = data['CT_samples_bwMetal']
MARpara = get_mar_params(config['mar_dir'])

data_list = np.loadtxt(config['data_list'], dtype=str)

# Generate MAR data
for phase in splits:
    phase_dir = os.path.join(config['dataset_dir'], phase)

    image_indices = CTpara[f'{phase}_indices']
    mask_indices = CTpara[f'{phase}_mask_indices']

    image_size = [CTpara['imPixNum'], CTpara['imPixNum'], len(mask_indices)]
    sinogram_size = [CTpara['sinogram_size_x'], CTpara['sinogram_size_y'], len(mask_indices)]

    # prepare metal masks
    print("Preparing metal masks...")
    selected_metal = metal_masks[:, :, mask_indices]
    mask_all = np.zeros(image_size, dtype=np.single)
    metal_trace_all = np.zeros(sinogram_size, dtype=np.single)

    for ii in range(len(mask_indices)):
        mask_resize = resize(selected_metal[:, :, ii], (CTpara['imPixNum'], CTpara['imPixNum']), anti_aliasing=True)

        fan_beam_func = lambda x: fanbeam(x, CTpara['SOD'], FanSensorGeometry='arc',
                                          FanSensorSpacing=CTpara['angSize'],
                                          FanRotationIncrement=360/CTpara['angNum'])
        mask_proj = fan_beam_func(mask_resize)
        metal_trace = np.where(mask_proj > 0, 1, 0).astype(np.single)

        mask_all[:, :, ii] = mask_resize.T
        metal_trace_all[:, :, ii] = metal_trace.T

    for ii in range(len(image_indices)):
        image_name = data_list[image_indices[ii]]
        output_dir = os.path.join(phase_dir, image_name[:-4])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        gt_file = os.path.join(output_dir, 'gt.mat')
        if os.path.isfile(gt_file):
            continue

        print(f"[{phase}][{ii+1}/{len(image_indices)}] Processing {image_name}")
        raw_image = imread(os.path.join(config['raw_dir'], image_name))

        image = np.single(raw_image) - 32768
        image = resize(image, (CTpara['imPixNum'], CTpara['imPixNum']), anti_aliasing=True)
        image[image < -1000] = -1000

        ma_sinogram_all, LI_sinogram_all, poly_sinogram, ma_CT_all, LI_CT_all, poly_CT, gt_CT, metal_trace_all = simulate_metal_artifact(
            image, selected_metal, CTpara, MARpara)

        np.savez(gt_file, image=image.astype(np.uint16))

        for jj in range(ma_CT_all.shape[2]):
            ct_file = os.path.join(output_dir, f"{jj}.mat")
            current_image = ma_CT_all[:, :, jj]
            np.savez(ct_file, image=current_image.astype(np.uint16))