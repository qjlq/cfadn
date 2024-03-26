import math
import os
import yaml
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# from helper import get_mar_params, simulate_metal_artifact
from cfadn import helper
import tomopy
import numpy as np


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


        def fanbeam(image, SOD, FanSensorGeometry='arc', FanSensorSpacing=None, FanRotationIncrement=None):
            # 假设image是2D CT切片，SOD是源到物体的距离
            # 其他参数FanSensorGeometry、FanSensorSpacing和FanRotationIncrement需要与CT扫描设置匹配

            # 根据实际情况设置这些值
            num_angles = 360  # 扫描角度数量
            det_count = image.shape[1]  # 探测器像素数量，假设图像为行：探测器列数，列：像素高度
            rotation_axis = 'vertical'  # 假设旋转轴垂直于床面

            if FanSensorGeometry == 'arc':
                # 对于弧形排列的探测器
                if FanSensorSpacing is None:
                    raise ValueError("For arc FanSensorGeometry, FanSensorSpacing must be provided.")
                ang = np.linspace(0, 360 - FanRotationIncrement, num=num_angles, endpoint=False)
                sino, _ = tomopy.project(image, angl=ang, center=det_count // 2, proj_type='平行束')

            elif FanSensorGeometry == 'linear':
                # 对于线性排列的探测器，此处假设探测器间距与FanSensorSpacing一致
                if FanSensorSpacing is None or FanRotationIncrement is None:
                    raise ValueError(
                        "For linear FanSensorGeometry, both FanSensorSpacing and FanRotationIncrement must be provided.")
                angles = np.arange(num_angles) * FanRotationIncrement
                sino, _ = tomopy.project(image, theta=angles, pixel_size_x=FanSensorSpacing,
                                         pixel_size_y=FanSensorSpacing,
                                         dist_source_detector=SOD, axis=rotation_axis)

            else:
                raise ValueError(f"Unsupported FanSensorGeometry: {FanSensorGeometry}")

            return sino


        # 使用函数的例子：
        # 假设我们有一个2D图像 data2d 和一组角度 angles
        # data2d = np.random.rand(256, 256)  # 示例数据
        # angles = np.linspace(0, 360, num=360, endpoint=False)  # 角度范围从0到360度，共360个均匀分布的角度
        # sod = 980.0
        # pixel_spacing = 0.5
        # margin = 10  # 假设的探测器边缘到有效区域的距离
        #
        # proj = fanbeam_projection(data2d, sod, angles[1] - angles[0], pixel_spacing, CTpara['sinogram_size_y'])

        # 注意：此简化版本可能不适用于所有情况，实际应用时应使用准确的系统参数进行计算。


        # fan_beam_func = lambda x: fanbeam_projection(x, CTpara['SOD'], CTpara['angNum'],
        #                                   CTpara['angSize'],CTpara['sinogram_size_y'])

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