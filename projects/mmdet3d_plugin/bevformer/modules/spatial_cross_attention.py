
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import pdb
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 use_weight=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.use_weight = use_weight
        self.use_linear=True
        if self.use_weight:
            self.pose_embedding = nn.Sequential(
                nn.Linear(16,embed_dims),
                #nn.Sigmoid(),
            )
            if self.use_linear:
                self.select_offsets=nn.Sequential(
                    nn.Linear(embed_dims+embed_dims, 128),
                    nn.Linear(128,1),
                    nn.Sigmoid(),
                )
            else:
                self.select_offsets= nn.Sequential(
                    nn.Conv2d(in_channels=embed_dims*2, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.Linear(128,1),
                    nn.Sigmoid(),
                    )

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if self.use_weight:
            constant_init(self.select_offsets, 1.)
            constant_init(self.pose_embedding, val=0., bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                bev_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                test=False,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        #7.28 multiply sigmoid weight to softmaxed_attention weights.
        #new attention weights=\sigma_{num_query,num_cams}new_offsets(Q_not_rebatched,trans_matrix)*attention weights
        lidar2img= kwargs['img_metas'][0]['lidar2img']
        lidar2img = np.asarray(lidar2img)
        
        lidar2img = reference_points.new_tensor(lidar2img)
        num_cams_all=lidar2img .shape[0]
        
        if self.use_weight:
            
            position_embed = self.pose_embedding(lidar2img.view(bs*num_cams_all,1,16)).repeat(1,num_query,1)
            
            # Concatenate position embeddings with input data
            input_with_position = torch.cat([(query+bev_pos).repeat(bs*num_cams_all,1,1), 10*F.normalize(position_embed,dim=2)], dim=-1)
            if self.use_linear:
                select_weight=self.select_offsets(input_with_position)
            
            #9.7 try conv
            else:
                input_size=input_with_position.size()
                h_sqrt = int(input_size[1] ** 0.5)
                input_with_position=input_with_position.view(input_size[0], input_size[2], h_sqrt, h_sqrt)
                
                output_tensor =self.select_offsets(input_with_position)
                
                output_size = output_tensor.size()
                #pdb.set_trace()
                select_weight = output_tensor.view(output_size[0], output_size[2]*output_size[3], output_size[1])

            # 8.12 3 tests for normalization:
            # 1 clamp
            #input_with_position = torch.cat([(query+bev_pos).repeat(bs*num_cams_all,1,1), F.normalize(position_embed,dim=2)], dim=-1)
            
            # 2 1/sqrt(d)
            #input_with_position = torch.cat([(query+bev_pos).repeat(bs*num_cams_all,1,1), position_embed/16], dim=-1)
            
            # 3 cos(bev,pose_emb)
            # 4 conv for concated feature: for neighbor bev infomation
            #clamped_tensor = torch.clamp(tensor, min=-1, max=1)

            
            #pdb.set_trace()
        else:
            select_weight=  torch.ones([bs*num_cams_all,num_query,1]).cuda()
            #attention_weights=torch.mul(select_weight.unsqueeze(-1),attention_weights)
        #pdb.set_trace()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])
        #print('\norigin',[len(each) for each in indexes])
        #print([len(each) f,or each in indexes],sum([len(each) for each in indexes]))
        original_num_query=sum([len(each) for each in indexes])
        if self.use_weight and test:
            indexes = []
            select_weight_as_mask=select_weight>1e-2
            for i, mask_per_img in enumerate(bev_mask):#12,1,2500,4
                mask_per_img=mask_per_img *select_weight_as_mask[i]
                index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                indexes.append(index_query_per_img)
            max_len = max([len(each) for each in indexes])
            #print('\nnew',[len(each) for each in indexes])
            #print([len(each) for each in indexes],sum([len(each) for each in indexes]))
            new_num_query=sum([len(each) for each in indexes])
            #print(new_num_query/original_num_query)
            #pdb.set_trace()
        num_cams, l, bs, embed_dims = key.shape
        num_cams=num_cams_all
        #print([len(each) for each in indexes])
        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, num_cams, max_len, D, 2])
        select_weight_rebatch =select_weight.new_zeros([bs,num_cams, max_len, 1])
        #print( select_weight.mean(),[len(each) for each in indexes])
        #pdb.set_trace()# 两件事 一个是pos_emb是none，另一个是concat的没有norm 8.6
        #print(num_cams_all)
        weight_count=0 
        weight_mean=0
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                #pdb.set_trace()
                
                select_weight_rebatch[j, i, :len(index_query_per_img)]=select_weight[(j+1)*i, index_query_per_img]
                weight_mean+=select_weight[(j+1)*i, index_query_per_img].sum()
                weight_count  +=select_weight[(j+1)*i, index_query_per_img].shape[0]
        #print(key[:num_cams_all].shape)         
        key = key[:num_cams_all].permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, self.embed_dims)
        value = value[:num_cams_all].permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, self.embed_dims)
        #print(select_weight.shape)
        def save_npy(number):
    
            np.save('bev_mask{}'.format(number),bev_mask.cpu())
            np.save('weight{}'.format(number),select_weight.cpu())
            np.save('reference_pts{}'.format(number),reference_points_cam.cpu())
            np.save('kwargs{}'.format(number),kwargs)
        #
        #print(select_weight[:6].mean())
        #kwargs   ['img_metas'][0]
        #reference_points_cam ([12, 1, 2500, 4, 2])
        #bev_mask
        #visual_weights(select_weight[0].reshape(50,50).cpu().numpy(),lidar2img[0].cpu().numpy())
        queries = self.deformable_attention(query=queries_rebatch.view(bs*num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index,select_weight=select_weight_rebatch.view(bs*num_cams, max_len,1),**kwargs).view(bs, num_cams, max_len, self.embed_dims)
        select_weight_r=select_weight_rebatch
        if self.use_weight and test:
            #num_cams, l, bs, embed_dims = key.shape
            num_cams=num_cams_all
            max_len1 = max([len(each) for each in indexes][:6])
            if num_cams>6:
                max_len2 = max([len(each) for each in indexes][6:])
            #print([len(each) for each in indexes])
            #print(max_len1, max_len2)
            #print([len(each) for each in indexes])
            # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
            queries_rebatch = query.new_zeros(
                [bs, 6, max_len1, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs, 6, max_len1, D, 2])
            select_weight_rebatch =select_weight.new_zeros([bs,6, max_len1, 1])
            for j in range(bs):
                for i, reference_points_per_img in enumerate(reference_points_cam[:6]):   
                    index_query_per_img = indexes[i]
                    queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                    #pdb.set_trace()
                    
                    select_weight_rebatch[j, i, :len(index_query_per_img)]=select_weight[(j+1)*i, index_query_per_img]
                    weight_mean+=select_weight[(j+1)*i, index_query_per_img].sum()
                    weight_count  +=select_weight[(j+1)*i, index_query_per_img].shape[0]
                queries1 = self.deformable_attention(query=queries_rebatch.view(bs*6, max_len1, self.embed_dims), key=key[:bs *6], value=value[:bs *6],
                                            reference_points=reference_points_rebatch.view(bs*6, max_len1, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index,select_weight=select_weight_rebatch.view(bs*6, max_len1,1),**kwargs).view(bs, 6, max_len1, self.embed_dims)
            if num_cams>6:
                queries_rebatch = query.new_zeros(
                    [bs, num_cams-6, max_len2, self.embed_dims])
                reference_points_rebatch = reference_points_cam.new_zeros(
                    [bs, num_cams-6, max_len2, D, 2])
                select_weight_rebatch =select_weight.new_zeros([bs,num_cams-6, max_len2, 1])
                    #print(select_weight_rebatch.shape)
                for j in range(bs):
                    for i, reference_points_per_img in enumerate(reference_points_cam[6:]):   
                        index_query_per_img = indexes[i+6]
                        queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                        reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                        #pdb.set_trace()
                        
                        select_weight_rebatch[j, i, :len(index_query_per_img)]=select_weight[(j+1)*(i+6), index_query_per_img]
                        weight_mean+=select_weight[(j+1)*(i+6), index_query_per_img].sum()
                        weight_count  +=select_weight[(j+1)*(i+6), index_query_per_img].shape[0]
                    queries2= self.deformable_attention(query=queries_rebatch.view(bs*(num_cams-6), max_len2, self.embed_dims), key=key[bs *6:], value=value[bs *6:],
                                                reference_points=reference_points_rebatch.view(bs*(num_cams-6), max_len2, D, 2), spatial_shapes=spatial_shapes,
                                                level_start_index=level_start_index,select_weight=select_weight_rebatch.view(bs*(num_cams-6), max_len2,1),**kwargs).view(bs,  num_cams-6, max_len2, self.embed_dims)
                
        if  self.use_weight and test:
            for j in range(bs):
                for i, index_query_per_img in enumerate(indexes):
                    if i<6:
                        slots[j, index_query_per_img] += queries1[j, i, :len(index_query_per_img)]
                    else:
                        slots[j, index_query_per_img] += queries2[j, i-6, :len(index_query_per_img)]
                    
        else:
            for j in range(bs):
                for i, index_query_per_img in enumerate(indexes):
                    slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
            
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        
        #print(weight_mean/weight_count)
        #pdb.set_trace()
        return self.dropout(slots) + inp_residual,select_weight_r


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.use_weight=False
        if self.use_weight:
            self.pose_embedding = nn.Linear(16,embed_dims)
            self.select_offsets=nn.Sequential(
                nn.Linear(embed_dims+embed_dims, 1),
                nn.Sigmoid(),
            )
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        if self.use_weight:
            constant_init(self.select_offsets, 1.)
            constant_init(self.pose_embedding, val=0., bias=0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                select_weight=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        lidar2img= kwargs['img_metas'][0]['lidar2img']
        lidar2img = np.asarray(lidar2img)
        
        lidar2img = reference_points.new_tensor(lidar2img)
        #
        
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        attention_weights=torch.multiply(select_weight.view(bs, num_query,1,1,1),attention_weights)
        #print(select_weight.shape)
        #pdb.set_trace()
        '''if self.use_weight:
            num_query_all=bev_query.shape[1]
            position_embed = self.pose_embedding(lidar2img.view(bs,1,16)).repeat(1,num_query_all,1)
            
            num_cams_all=int(bs/bev_query.shape[0])
            # Concatenate position embeddings with input data
            input_with_position = torch.cat([bev_query.repeat(num_cams_all,1,1), position_embed], dim=-1)
            select_weight=self.select_offsets(input_with_position)
            #pdb.set_trace()
            attention_weights=torch.mul(select_weight.unsqueeze(-1),attention_weights)
        '''
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

