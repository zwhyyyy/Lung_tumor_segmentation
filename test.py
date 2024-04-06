# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.nn.functional as F
import json
from segment_anything_volumetric import sam_model_registry
from network.model import SegVol
from data_process.demo_data_process import process_ct_gt
from data_process.demo_data_process import process_ct_gt_
import monai.transforms as transforms
from utils.monai_inferers_utils import sliding_window_inference, generate_box, select_points, build_binary_cube, build_binary_points, logits2roi_coor
from utils.visualize import draw_result
from utils.visualize import draw_result_

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", default=True, type=bool)
    parser.add_argument("--resume", type = str, default = './preTrain/medsam_model_e440.pth')  # ./preTrain/medsam_model_e300.pth
    parser.add_argument("-infer_overlap", default=0.5, type=float, help="sliding window inference overlap")  # 重叠比例
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple)  # 期望的空间尺寸 D为32 就是沿着Z轴的切分数  因为图片是3D的
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple)  # 就是把图像划分为多少块
    parser.add_argument('-work_dir', type=str, default='./results')
    parser.add_argument("--clip_ckpt", type = str, default = './config/clip')
    args = parser.parse_args(args=[])
    return args





def zoom_in_zoom_out(args, segvol_model, image, image_resize, zuobiao=None, categories=["lung tumor"], ifbox = True):
    args.use_box_prompt = ifbox
    logits_labels_record = {}
    image = image.unsqueeze(0)
    image_single_resize = image_resize.unsqueeze(0)  # 缩小的图像
    image_single = image[0, 0]  # 传统transform中的图像
    ori_shape = image_single.shape

    for item_idx in range(1):  # 这里也需要改
        text_single = 'lung tumor'
        points_single = None
        box_single = None

        if args.use_box_prompt:  # 在这里 得到一个1行6列的tensor矩阵
            box_single = torch.unsqueeze(torch.tensor(zuobiao, device='cuda:0'), 0)
            shape = torch.Size([32, 256, 256])
            binary_cube_resize = build_binary_cube(box_single, binary_cube_shape=shape)  # 生成立体形状

        ####################
        # zoom-out inference:  缩小推理范围
        with torch.no_grad():
            logits_global_single = segvol_model(image_single_resize.cuda(),
                                                text=text_single,
                                                boxes=box_single,
                                                points=points_single) # 这个就是反卷积之后的 解码之后向量 （1，1，32，256，256）

        logits_global_single = F.interpolate(
            logits_global_single.cpu(),
            size=ori_shape, mode='nearest')[0][0]  # ori_shape（304，476，512）为此时维度为（270，423，512）


        if args.use_box_prompt:
            binary_cube = F.interpolate(  # 把格子向量 映射成原图形状的
                binary_cube_resize.unsqueeze(0).unsqueeze(0).float(),
                size=ori_shape, mode='nearest')[0][0]

        logits_labels_record[categories[item_idx]] = (
            image_single,
            points_single,
            box_single,
            logits_global_single)

        ####################
        # zoom-in inference:  # 放大推理

        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(args.spatial_size, logits_global_single)
        # Crop roi
        image_single_cropped = image_single[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1].unsqueeze(0).unsqueeze(0)
        global_preds = (torch.sigmoid(logits_global_single[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]) > 0.5).long()  # 对裁剪后的数据进行推理
        prompt_reflection = None
        if args.use_box_prompt:
            binary_cube_cropped = binary_cube[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_reflection = (
                binary_cube_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0)
            )


        with torch.no_grad():
            logits_single_cropped = sliding_window_inference(
                    image_single_cropped.cuda(), prompt_reflection,
                    args.spatial_size, 1, segvol_model, args.infer_overlap,
                    text=text_single,
                    use_box=args.use_box_prompt,
                    use_point=args.use_point_prompt,
                )   # 滑动窗口进行预测 得到最后预测好的图片向量
            logits_single_cropped = logits_single_cropped.cpu().squeeze()

        logits_global_single[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1] = logits_single_cropped
        logits_labels_record[categories[item_idx]] = (
            image_single,  # 这个是原图像
            points_single,
            box_single,
            logits_global_single,)  # 这个是预测结果 过一下激活函数
        print('完成了')
    return logits_labels_record





def inference_single_ct(args, segvol_model, data_item, categories):
    segvol_model.eval()  # 下边这也需要改
    image = data_item["image"].float()  # 放入传统的transforms中
    image_zoom_out = data_item["zoom_out_image"].float()  # 缩小的图像
    visualization_image = data_item['visualization']   # 就是用这个显示图片的
    visualization_image = visualization_image.detach().cpu().numpy()
    visualization_image = visualization_image * 255
    return visualization_image, image, image_zoom_out


def main(args, filename):
    gpu = 0
    torch.cuda.set_device(gpu)
    # build model
    sam_model = sam_model_registry['vit'](args=args)   # 定义了自己的sam对象
    segvol_model = SegVol(
                        image_encoder=sam_model.image_encoder,  # image_encoder mask_decoder prompt_encoder 都是用的sam的
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,   # 不管是编码之后 点 框 文本 都是256维向量
                        clip_ckpt=args.clip_ckpt,
                        roi_size=args.spatial_size,  # 期望的空间尺寸 因为图片是3D的
                        patch_size=args.patch_size,   # 就是把图像划分小块的尺寸
                        test_mode=args.test_mode,
                        ).cuda()  # 定义了自己的SegVol对象
    segvol_model = torch.nn.DataParallel(segvol_model, device_ids=[gpu])   # 放到GPU上去跑

    # load param   加载预训练模型
    if os.path.isfile(args.resume):
        ## Map model to be loaded to specified single GPU
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        segvol_model.load_state_dict(checkpoint['model'], strict=True)   # load_state_dict是nn.Module对象
        print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    categories = ["lung tumor"]
    ct_path = filename
    data_item = process_ct_gt_(ct_path, categories, args.spatial_size)

    # seg config for prompt & zoom-in-zoom-out   这里需要实时变换
    args.use_zoom_in = True
    args.use_text_prompt = True
    args.use_box_prompt = True
    args.use_point_prompt = False
    args.visualize = True
    return segvol_model, data_item, categories








if __name__ == "__main__":
    args = set_parse()
    main(args)