# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import torch
import monai.transforms as transforms
list_name = []
def draw_result_(image, logits, spatial_size):
    global list_name
    zoom_out_transform = transforms.Compose([
        transforms.AddChanneld(keys=["image", "logits"]),  # 从这里移除 'label'
        transforms.Resized(keys=["image", "logits"], spatial_size=spatial_size, mode='nearest-exact')
    ])
    post_item = zoom_out_transform({
        'image': image,  # （304，476，512）
        'logits': logits
    })
    image,logits = post_item['image'][0],post_item['logits'][0]
    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).int()  # 看其有多少个像素点 然后成比例
    root_dir = os.path.join("./results", f'fig_examples/{"tests"}/')
    root_dir = "./static/img2"
    # if not os.path.exists(root_dir):
    #     os.makedirs(root_dir)
    total_voxel = 0
    for j in range(image.shape[0]):
        img_2d = image[j, :, :].detach().cpu().numpy()
        preds_2d = preds[j, :, :].detach().cpu().numpy()
        print(f"{j}:np.sum(preds_2d) = {np.sum(preds_2d)}")
        if np.sum(preds_2d) == 0:  # 必须都不是0 才能继续往下走
            continue
        total_voxel = total_voxel + np.sum(preds_2d)  # 这里的total_voxel要传到接口
        img_2d = img_2d * 255
        # orginal img
        fig, ax1 = plt.subplots(1, 1)  # imshow这个就可以显示图片
        # ax1.imshow(img_2d, cmap='gray')
        # ax1.set_title('Image with prompt')
        # ax1.axis('off')

        # preds
        ax1.imshow(img_2d, cmap='gray')
        show_mask(preds_2d, ax1)
        ax1.set_title('Prediction')
        ax1.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(root_dir, f'{"test"}_{j}.png'), bbox_inches='tight')
        list_name.append(os.path.join(root_dir, f'{"test"}_{j}.png'))
        plt.close()
    # 把要保存的值写入文件
    write_var_to_file(total_voxel)


def draw_result(category, image, bboxes, points, logits, gt3D, spatial_size, work_dir):
    zoom_out_transform = transforms.Compose([
        transforms.AddChanneld(keys=["image", "label", "logits"]),   # 添加一个新的通道维度
        # 将这三个键对应的数据调整到spatial_size指定的大小
        transforms.Resized(keys=["image", "label", "logits"], spatial_size=spatial_size, mode='nearest-exact')
    ])
    post_item = zoom_out_transform({
        'image': image,  # （304，476，512）
        'label': gt3D,
        'logits': logits
    })  # （1，32，32，256） 走完上边代码之后  它们的维度都变成右边的维度了
    image, gt3D, logits = post_item['image'][0], post_item['label'][0], post_item['logits'][0]
    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).int()

    root_dir=os.path.join(work_dir, f'fig_examples/{category}/') 

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if bboxes is not None:
        x1, y1, z1, x2, y2, z2 = bboxes[0].cpu().numpy()
    if points is not None:
        points = (points[0].cpu().numpy(), points[1].cpu().numpy())
        points_ax = points[0][0]   # [n, 3]
        points_label = points[1][0] # [n]

    for j in range(image.shape[0]):
        img_2d = image[j, :, :].detach().cpu().numpy()
        preds_2d = preds[j, :, :].detach().cpu().numpy()
        label_2d = gt3D[j, :, :].detach().cpu().numpy()
        print(f"{j}:  np.sum(label_2d) = {np.sum(label_2d)},np.sum(preds_2d) = {np.sum(preds_2d)}")
        print()
        if np.sum(label_2d) == 0 or np.sum(preds_2d) == 0:  # 必须都不是0 才能继续往下走
            continue

        img_2d = img_2d * 255
        # orginal img
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)   # imshow这个就可以显示图片
        ax1.imshow(img_2d, cmap='gray')
        ax1.set_title('Image with prompt') 
        ax1.axis('off')

        # gt
        ax2.imshow(img_2d, cmap='gray')
        show_mask(label_2d, ax2)
        ax2.set_title('Ground truth') 
        ax2.axis('off')

        # preds
        ax3.imshow(img_2d, cmap='gray')
        show_mask(preds_2d, ax3)
        ax3.set_title('Prediction') 
        ax3.axis('off')

        # boxes
        if bboxes is not None:
            if j >= x1 and j <= x2:
                show_box((z1, y1, z2, y2), ax1)
        # points
        if points is not None:
            for point_idx in range(points_label.shape[0]):
                point = points_ax[point_idx]
                label = points_label[point_idx] # [1]
                if j == point[0]:
                    show_points(point, label, ax1)
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(root_dir, f'{category}_{j}.png'), bbox_inches='tight')
        plt.close()

def show_mask(mask, ax):
    color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]  # 这是取最后两个维度呢
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=0.35)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def show_points(points_ax, points_label, ax):
    color = 'red' if points_label == 0 else 'blue'
    ax.scatter(points_ax[2], points_ax[1], c=color, marker='o', s=200)



def write_var_to_file(value):
    with open('./parameter/var.txt', 'w') as f:
        f.write(str(value))