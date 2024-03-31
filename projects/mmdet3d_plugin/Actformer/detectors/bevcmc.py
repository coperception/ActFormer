# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by hsz
# ---------------------------------------------

import pdb
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d

from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVCMC(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 use_LiDAR=True,
                 use_Cam=True,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVCMC,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.use_LiDAR = use_LiDAR
        self.use_Cam = use_Cam

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        #pdb.set_trace()
        x = self.pts_middle_encoder(voxel_features, coors, batch_size.item())

        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(
                    int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img','points'))
    def extract_feat(self, points, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        if self.use_Cam:
            img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = None

        if self.use_LiDAR:
            pts_feats = self.extract_pts_feat(points)
        else:
            pts_feats = None

        return (img_feats, pts_feats)

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, points_queue, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            #imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            
            
            
            #feats_list = self.extract_feat(points=points_queue, img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = self.extract_img_feat(img=imgs_queue[:,i], img_metas=img_metas)
                #img_feats list [0] torch.Size([b, 12, 256, 29, 50])
                '''points_per_queue=[]
                for b in range(bs):
                    points_per_queue.append(points_queue[b][i])
                #points_per_queue list [b][num_cars] [points]
                
                pts_feats=[]
                for b in range(bs):
                    pts_feats.append([])
                    # 5 cars cuda run time error on single GPU
                    pts_feats[b]=self.extract_pts_feat(pts=points_per_queue[b])
                '''    
        
                new_pts_feats=[]
                '''num_car=len(points_per_queue[0])
                for car in range(num_car):
                    new_pts_feats.append([])
                    for b in range(bs):
                        new_pts_feats[car].append(pts_feats[b][0][car])
                    new_pts_feats[car]=torch.stack(new_pts_feats[car])'''
                #pts_feats=self.extract_pts_feat(pts=points_per_queue)
                #pdb.set_trace()         
                #(img_feats, pts_feats) = [each_scale[:, i]for each_scale in feats_list]
                #pts_feats[num_cars] [b, 512, 128, 128]
                
                prev_bev = self.pts_bbox_head(new_pts_feats,
                                              img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        #img [1, 4,6, 3, 928, 1600]
        #points (2,4,each dim,5)
        #pdb.set_trace()
        bs=img.size(0)
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        
        #img [1, 6*num_car, 3, 928, 1600]
        
        #

        prev_img_metas = copy.deepcopy(img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        #dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 'pts_filename', 'scene_token', 'can_bus', 'prev_bev_exists'])
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        
        img_feats=self.extract_img_feat(img=img,img_metas=img_metas)
        #pdb.set_trace()
        #points list[bs,queue_len,num_car]
        
        use_lidar=False
        if use_lidar:
            num_car=len(points[0][0])
            present_points=[]
            prev_points=[]
            for b in range(bs):
                present_points.append(points[b][-1])
                prev_points.append(points[b][:-1])
            #present_points bs,num_car
            #prev_points bs,queue_len-1,num_car
            #pdb.set_trace()
            pts_feats=[]
            
            for b in range(bs):
                pts_feats.append([])
                # 5 cars cuda run time error on single GPU
                pts_feats[b]=self.extract_pts_feat(pts=present_points[b])
                
            
            #pts_feats list [bs][1][num_car] each:[ 512, 128, 128]
            #should change to [num_car]  each:[bs, 512, 128, 128]
            new_pts_feats=[]
            for car in range(num_car):
                new_pts_feats.append([])
                for b in range(bs):
                    new_pts_feats[car].append(pts_feats[b][0][car])
                new_pts_feats[car]=torch.stack(new_pts_feats[car])
            #pdb.set_trace()         
                    
            
            #prev_points [bs,queue_legnth-1] each
        if not use_lidar:
            prev_bev = self.obtain_history_bev(
                None, prev_img, prev_img_metas)
            losses = dict()
            #pdb.set_trace()
            #!pts_feats input:list of num_car, each [bs,C,h0,w0]
            losses_pts = self.forward_pts_train(None, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        #pdb.set_trace()
        return losses

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        #pdb.set_trace()
        #print(img_metas[0][0])#['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow'])
        #? no canbus or scene token
        #print(len(img))#img[0].shapetorch.Size([1, 12, 3, 928, 1600])

        #print(points)points list [num_cars,b,1]
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        
        #dataset里面 train data 做了处理 这里是test 
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][0][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][0][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            
            img_metas[0][0]['can_bus'][0][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][0][-1] -= self.prev_frame_info['prev_angle']
            for j in range(1,len(img_metas[0][0]['can_bus'])):
                img_metas[0][0]['can_bus'][j][-1] -= tmp_angle 
                img_metas[0][0]['can_bus'][j][:3] -= tmp_pos
            
        else:
            for j in range(len(img_metas[0][0]['can_bus'])):
                img_metas[0][0]['can_bus'][j][-1] -= tmp_angle 
                img_metas[0][0]['can_bus'][j][:3] -= tmp_pos

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, pts_feats,
                        img_feats, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(pts_feats,
                                  img_feats, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        #pdb.set_trace()
        #what is points shape?
        img_feats=self.extract_img_feat(img=img,img_metas=img_metas)
        pts_feats=[]
        
        '''for car in range(len(points)):
            for b in range(len(points[0])):
                points[car][b]=points[car][b][0]
            pts_feats.append([])
            # 5 cars cuda run time error on single GPU
            pts_feats[car]=self.extract_pts_feat(pts=points[car])
            #pdb.set_trace()
            pts_feats[car]=torch.stack(pts_feats[car]).squeeze(1)
            #不确定是squeeze哪个dim
        '''
        '''if self.use_LiDAR:
            pts_feats = [feat.unsqueeze(dim=1) for feat in pts_feats]'''

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(pts_feats,
                                                      img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
