# -*- coding: utf-8 -*-
import numpy as np
import monai.transforms as transforms


class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d

class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d

class ForegroundNormalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            d[key] = self.normalize(d[key])
        return d
    
    def normalize(self, ct_narray):
        ct_voxel_ndarray = ct_narray.copy()
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        ### transform ###
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray

def process_ct_gt_(case_path, category, spatial_size):
    img_loader = transforms.LoadImage()
    transform = transforms.Compose(
        [
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            ForegroundNormalization(keys=["image"]),
            DimTranspose(keys=["image"]),
            MinMaxNormalization(),
            # 这个转换对图像进行空间填充(内容不变） 如果自己的图像比spatial_size大则不会填充
            transforms.SpatialPadd(keys=["image"], spatial_size=spatial_size, mode='constant'),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    zoom_out_transform = transforms.Resized(keys=["image"], spatial_size=spatial_size, mode='nearest-exact')
    show_image = transforms.Resized(keys=["image"], spatial_size=(256, 256, 256), mode='nearest-exact')
    item = {}
    # generate ct_voxel_ndarray （512，512，304）
    ct_voxel_ndarray, _ = img_loader(case_path)  # 导入预测图像 ct_voxel_ndarray ndarray格式
    ct_voxel_ndarray = np.array(ct_voxel_ndarray).squeeze()  # .squeeze方法是移除数组形状中所有单维度的条目
    ct_shape = ct_voxel_ndarray.shape
    total_voxels = ct_shape[0] * ct_shape[1] * ct_shape[2]
    write_var_to_file(total_voxels)
    ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray,axis=0)  # expand_dims指定的轴位置增加一个新的维度  这里是0 就是第一维度 也就是（1，512，512，270）
    item['image'] = ct_voxel_ndarray
    item = transform(item)  # 先经过transform 再经过zoom_out_transform  经过此处时图像的维度也发生了变化
    item_zoom_out = zoom_out_transform(item)
    show_visualization = show_image(item)

    visualization_image = show_visualization['image']
    visualization_image = visualization_image.squeeze()  # 这个矩阵就是可视化展示的
    item['visualization'] = visualization_image
    item['zoom_out_image'] = item_zoom_out['image']
    return item







def process_ct_gt(case_path, label_path, category, spatial_size):
    print('Data preprocessing...')
    # transform  读入数据用到了 monai包
    img_loader = transforms.LoadImage()
    transform = transforms.Compose(
        [
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            ForegroundNormalization(keys=["image"]),
            DimTranspose(keys=["image", "label"]),
            MinMaxNormalization(),
            # 这个转换对图像和标签进行空间填充(内容不变） 如果自己的图像比spatial_size大则不会填充
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, mode='constant'),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    # Resized把图像指定成指定大小 （内容可能会变化）根据插值或者下采样
    zoom_out_transform = transforms.Resized(keys=["image", "label"], spatial_size=spatial_size, mode='nearest-exact')

    item = {}
    # generate ct_voxel_ndarray （512，512，270）
    ct_voxel_ndarray, _ = img_loader(case_path)  # 导入预测图像 ct_voxel_ndarray ndarray格式
    print(type(ct_voxel_ndarray))
    ct_voxel_ndarray = np.array(ct_voxel_ndarray).squeeze()  # .squeeze方法是移除数组形状中所有单维度的条目
    ct_shape = ct_voxel_ndarray.shape
    ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0) # expand_dims指定的轴位置增加一个新的维度  这里是0 就是第一维度 也就是（1，512，512，270）
    item['image'] = ct_voxel_ndarray

    # generate gt_voxel_ndarray （512，512，304）  # 这一段不该有
    gt_voxel_ndarray, _ = img_loader(label_path)
    gt_voxel_ndarray = np.array(gt_voxel_ndarray)
    present_categories = np.unique(gt_voxel_ndarray)
    gt_masks = []  # 为category每个部分的背景和实体category有几个len就为几
    for cls_idx in range(len(category)):
        # ignore background  把其划分为四个labels 即一个categories为一个label 其它的为背景
        cls = cls_idx + 1
        if cls not in present_categories:
            gt_voxel_ndarray_category = np.zeros(ct_shape)
            gt_masks.append(gt_voxel_ndarray_category)
        else:
            gt_voxel_ndarray_category = gt_voxel_ndarray.copy()  # 也就是标签值
            gt_voxel_ndarray_category[gt_voxel_ndarray != cls] = 0
            gt_voxel_ndarray_category[gt_voxel_ndarray == cls] = 1
            gt_masks.append(gt_voxel_ndarray_category)
    gt_voxel_ndarray = np.stack(gt_masks, axis=0)  # （1，512，512，304）
    assert gt_voxel_ndarray.shape[0] == len(category) and gt_voxel_ndarray.shape[1:] == ct_voxel_ndarray.shape[1:]
    item['label'] = gt_voxel_ndarray.astype(np.int32)  # 在此刻 item['image']:(1,512,512,270)  item['label']:(len(category),512,512,270)

    # transform （1，270，423，512）   （4，270，423，512）
    item = transform(item)  # 先经过transform 再经过zoom_out_transform  经过此处时图像的维度也发生了变化
    item_zoom_out = zoom_out_transform(item)
    item['zoom_out_image'] = item_zoom_out['image']
    item['zoom_out_label'] = item_zoom_out['label']
    print(  'Zoom_in image shape: ',   item['image'].shape, 
          '\nZoom_in label shape: ', item['label'].shape,
          '\nZoom_out image shape: ', item['zoom_out_image'].shape,
          '\nZoom_out label shape: ', item['zoom_out_label'].shape,
          )
    return item


def write_var_to_file(value):
    with open('./parameter/total_voxels.txt', 'w') as f:
        f.write(str(value))
