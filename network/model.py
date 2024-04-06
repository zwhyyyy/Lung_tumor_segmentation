# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
import random
from utils.monai_inferers_utils import select_points, generate_box
from utils.loss import BCELoss, BinaryDiceLoss
from torch.cuda.amp import autocast

#%% set up model
class SegVol(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                clip_ckpt,
                roi_size,
                patch_size,
                test_mode=False,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.text_encoder = TextEncoder(clip_ckpt)
        self.feat_shape = np.array(roi_size)/np.array(patch_size)  # feat_shape = spatial/patch 计算出每个小块特征图的期望尺寸
        self.test_mode = test_mode
        self.dice_loss = BinaryDiceLoss().cuda()
        self.bce_loss = BCELoss().cuda()
        self.decoder_iter = 6

    def forward(self, image, text=None, boxes=None, points=None, **kwargs):
        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])  # 因为一上来是五维度的 减去两个维度 传进去的就是经过变形的图片
        image_embedding, _ = self.image_encoder(image)  # vit （1 ，2048，768）
        image_embedding = image_embedding.transpose(1, 2).view(bs, -1,   # 重新确定形状 （1， 768， 8， 16， 16）
            int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))
        # test mode
        if self.test_mode:
            return self.forward_decoder(image_embedding, img_shape, text, boxes, points)
        
        # train mode
        ## sl
        sl_loss = self.supervised_forward(image, image_embedding, img_shape, kwargs['train_organs'], kwargs['train_labels'])
        ## ssl
        ssl_loss = self.unsupervised_forward(image, image_embedding, kwargs['pseudo_seg_cleaned'], img_shape)
        return sl_loss, ssl_loss

    def forward_decoder(self, image_embedding, img_shape, text=None, boxes=None, points=None):
        with torch.no_grad():
            if boxes is not None:
                if len(boxes.shape) == 2:
                    boxes = boxes[:, None, :] # (B, 1, 6)
            if text is not None:
                text_embedding = self.text_encoder(text)  # clip的  # (B, 768)
            else:
                text_embedding = None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(  # sam的
            points=points,
            boxes=boxes,
            masks=None,
            text_embedding=text_embedding,
        ) # 这里 （1，3，768）sparse_embeddings

        dense_pe = self.prompt_encoder.get_dense_pe()  # 返回位置编码
        low_res_masks, _ = self.mask_decoder(  # sam的 （1，1，32，64，64）
            image_embeddings=image_embedding,  # 图像编码
            text_embedding = text_embedding,  # 文本编码
            image_pe=dense_pe,  # 位置编码
            sparse_prompt_embeddings=sparse_embeddings,   #  稀疏空间编码
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
          )
        logits = F.interpolate(low_res_masks, size=img_shape, mode='trilinear', align_corners=False)  # 这一步就是 反卷积 对数据进行上采样或下采样
        return logits

    def supervised_forward(self, image, image_embedding, img_shape, training_organs, train_labels):
        iter_points, iter_bboxes, iter_organs = self.build_prompt_label(image.shape[0], training_organs, train_labels)
        # select prompt
        prompt_options = [[None, iter_points, iter_organs], [iter_bboxes, None, iter_organs], 
                        [None, None, iter_organs], [iter_bboxes, None, None], [None, iter_points, None],
                        [iter_bboxes, iter_points, None]]  # 分六种情况训练
        sl_loss = 0
        for prompt in prompt_options:
            bboxes, points, organs = prompt
            logits = self.forward_decoder(image_embedding, img_shape, text=organs, boxes=bboxes, points=points)
            # cal loss
            sl_loss_dice = self.dice_loss.forward(logits.squeeze().float(), train_labels.squeeze().float())
            sl_loss_bce = self.bce_loss.forward(logits.squeeze().float(), train_labels.squeeze().float())
            sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss
    
    def unsupervised_forward(self, image, image_embedding, pseudo_seg_cleaned, img_shape):
        sll_loss = 0
        for iter in range(self.decoder_iter):
            if iter % 2 == 0:
                pseudo_labels, pseudo_points_prompt = self.build_pseudo_point_prompt_label(image.shape, pseudo_seg_cleaned)
                logits = self.forward_decoder(image_embedding, img_shape, text=None, boxes=None, points=pseudo_points_prompt)
            else:
                pseudo_labels, pseudo_bboxes_prompt = self.build_pseudo_box_prompt_label(image.shape, pseudo_seg_cleaned)
                logits = self.forward_decoder(image_embedding, img_shape, text=None, boxes=pseudo_bboxes_prompt, points=None)
            # cal loss
            sll_loss_dice = self.dice_loss.forward(logits.squeeze().float(), pseudo_labels.squeeze().float())
            sll_loss_bce = self.bce_loss.forward(logits.squeeze().float(), pseudo_labels.squeeze().float())
            sll_loss += sll_loss_dice + sll_loss_bce
        return sll_loss

    def build_prompt_label(self, bs, training_organs, train_labels):
        # generate prompt & label
        iter_organs = []
        iter_bboxes = []
        iter_points_ax = []
        iter_point_labels = []
        for sample_idx in range(bs):
            # organ prompt
            iter_organs.append(training_organs)
            # box prompt
            box = generate_box(train_labels[sample_idx])
            iter_bboxes.append(box)
            # point prompt
            num_positive_extra_max, num_negative_extra_max = 10, 10
            num_positive_extra = random.randint(0, num_positive_extra_max)  # 随机生成点的数量
            num_negative_extra = random.randint(0, num_negative_extra_max)
            point, point_label = select_points(
                train_labels[sample_idx],
                num_positive_extra=num_positive_extra,
                num_negative_extra=num_negative_extra,
                fix_extra_point_num=num_positive_extra_max + num_negative_extra_max)
            iter_points_ax.append(point)
            iter_point_labels.append(point_label)
        # batched prompt
        iter_points_ax = torch.stack(iter_points_ax, dim=0).cuda()
        iter_point_labels = torch.stack(iter_point_labels, dim=0).cuda()
        iter_points = (iter_points_ax, iter_point_labels)
        iter_bboxes = torch.stack(iter_bboxes, dim=0).float().cuda()
        return iter_points, iter_bboxes, iter_organs
    
    def build_pseudo_point_prompt_label(self, input_shape, seg_labels):
        pseudo_labels = torch.zeros(input_shape).cuda()
        # generate points
        points = []
        point_labels = []
        for batch_idx in range(input_shape[0]):
            # generate pseudo label
            unique_ids = torch.unique(seg_labels[batch_idx])
            unique_ids = unique_ids[unique_ids != -1]
            region_id = random.choice(unique_ids).item()
            pseudo_labels[batch_idx][seg_labels[batch_idx]==region_id] = 1
            # generate point prompt
            num_positive_extra_max, num_negative_extra_max = 10, 10
            num_positive_extra = random.randint(4, num_positive_extra_max)
            num_negative_extra = random.randint(0, num_negative_extra_max)
            assert len(pseudo_labels[batch_idx][0].shape) == 3
            point, point_label = select_points(
                pseudo_labels[batch_idx][0],
                num_positive_extra=num_positive_extra,
                num_negative_extra=num_negative_extra,
                fix_extra_point_num=num_positive_extra_max + num_negative_extra_max)
            points.append(point)
            point_labels.append(point_label)
        points = torch.stack(points, dim=0).cuda()
        point_labels = torch.stack(point_labels, dim=0).cuda()
        pseudo_points_prompt = (points, point_labels)
        return pseudo_labels, pseudo_points_prompt

    def build_pseudo_box_prompt_label(self, input_shape, seg_labels_cleaned):
        pseudo_labels = torch.zeros(input_shape).cuda()
        iter_bboxes = []
        # generate boxes
        for batch_idx in range(input_shape[0]):
            # generate ori pseudo label
            unique_ids = torch.unique(seg_labels_cleaned[batch_idx])
            unique_ids = unique_ids[unique_ids != -1]
            region_id = random.choice(unique_ids).item()
            pseudo_labels[batch_idx][seg_labels_cleaned[batch_idx]==region_id] = 1
            # generate box prompt
            box = generate_box(pseudo_labels[batch_idx][0])
            iter_bboxes.append(box)
            # refine pseudo label
            x_min, y_min, z_min, x_max, y_max, z_max = box
            binary_cube = torch.zeros_like(pseudo_labels[batch_idx][0]).int()
            binary_cube[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1
            # cal iou
            mask_label = seg_labels_cleaned[batch_idx][0]
            assert binary_cube.shape == mask_label.shape, str(binary_cube.shape) + ' ' + str(mask_label.shape)
            mask_values_in_binary_cube = mask_label[binary_cube == 1]
            unique_mask_values = torch.unique(mask_values_in_binary_cube)
            # print('unique_mask_values ', unique_mask_values)
            for value in unique_mask_values:
                if value == -1: continue
                mask_area = (mask_label == value)
                intersection = (binary_cube & mask_area)
                iou = intersection.float().sum() / mask_area.float().sum()
                if iou > 0.90:
                    # print(f"Mask value {value} has IOU > 0.90 in binary cube.")
                    pseudo_labels[batch_idx][seg_labels_cleaned[batch_idx]==value] = 1

        bboxes = torch.stack(iter_bboxes, dim=0).float().cuda()
        return pseudo_labels, bboxes
    
class TextEncoder(nn.Module):
    def __init__(self, clip_ckpt):
        super().__init__()
        config = CLIPTextConfig()
        self.clip_text_model = CLIPTextModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_ckpt)
        self.dim_align = nn.Linear(512, 768)
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def organ2tokens(self, organ_names):
        text_list = ['A computerized tomography of a {}.'.format(organ_name) for organ_name in organ_names]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        for key in tokens.keys():
            tokens[key] = tokens[key].cuda()   # 利用tokenizer  input_ids 和 attention
        return tokens
    
    def forward(self, text):
        if text is None:
            return None
        if type(text) is str:
            text = [text]
        tokens = self.organ2tokens(text)
        clip_outputs = self.clip_text_model(**tokens)  # 对文本进行编码
        text_embedding = clip_outputs.pooler_output  # pooler_output通常是指模型输出的汇总表示
        text_embedding = self.dim_align(text_embedding)  # 线性层
        return text_embedding
