import copy
import pdb
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from functools import reduce


import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscenes_eval import NuScenesEval_custom
from .v2x_sim_eval import V2XSIMEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
from nuscenes.nuscenes import NuScenes as V2XSimDataset
# modified By Huang Suozhi


@DATASETS.register_module()
class CustomV2XSIMDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.is_single=False
        #self.nusc=V2XSimDataset('v1.0-mini', dataroot='data/V2X-Sim-2.0/', verbose=True)

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        #print(index_list)
        
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            #print(i)
            #print(i,self.nusc.get("scene",input_dict['scene_token'])["name"])
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
            #print(example['img'].data.shape) 
        #pdb.set_trace()
        #print(queue)
        return self.union2one(queue)

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][0][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][0][-1])
                metas_map[i]['can_bus'][0][:3] = 0
                metas_map[i]['can_bus'][0][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][0][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][0][-1])
                metas_map[i]['can_bus'][0][:3] -= prev_pos
                metas_map[i]['can_bus'][0][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        #print([len(each ) for each in imgs_list])18,18,18
        #print(len(imgs_list),len(metas_map)) 3,3 
        padded_imgs_list=[]
        for img in imgs_list:
            num_cams,N,H,W=img.shape
            #print(num_cams)
            if img.shape[0]>11:#only 12 for 2_cam
                #self.is_single=False
                
                
                padded_imgs_list.append(torch.cat((img,torch.zeros([30-num_cams,N,H,W])),dim=0))
            else:
                self.is_single=True
        #print(self.is_single,[len(each ) for each in imgs_list],[len(each ) for each in padded_imgs_list])
        if self.is_single==False:
            queue[-1]['img'] = DC(torch.stack(padded_imgs_list),
                              cpu_only=False, stack=True)
        else:
            queue[-1]['img'] = DC(torch.stack(imgs_list),
                                    cpu_only=False, stack=True)
        #print(imgs_list[0].shape,len(metas_map))#torch.Size([30, 3, 480, 800]) 3
        #print(len(metas_map[0]['lidar2img']))
        #test.py 不调用此代码
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        '''for ind in range(len(metas_map)):
            if len(metas_map[ind]['lidar2img'])!=18:
                print(len(metas_map[ind]['lidar2img']))'''
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            trans_matrix=[]
            #transform matrix
            #print(len(info['lidar2ego_translation']))
            #pdb.set_trace()
            for num_agent in range(len(info['lidar2ego_translation'])):
                
                # Homogeneous transform from ego car frame to reference frame
                ref_from_car = transform_matrix(
                    info['lidar2ego_translation'][0], Quaternion(info['lidar2ego_rotation'][0]), inverse=False
                )
                # Homogeneous transformation matrix from global to _current_ ego car frame
                car_from_global = transform_matrix(
                    info['ego2global_translation'][0], Quaternion(info['ego2global_rotation'][0]), inverse=False
                )
                if num_agent==0:
                    trans_matrix.append(np.eye(4))
                else:
                    global_from_car = transform_matrix(
                        info['ego2global_translation'][num_agent], Quaternion(info['ego2global_rotation'][num_agent]), inverse=True
                    )
                    car_from_current =  transform_matrix(
                        info['lidar2ego_translation'][num_agent], Quaternion(info['lidar2ego_rotation'][num_agent]), inverse=True
                    )
                    trans_matrix.append( reduce(
                    np.dot, [car_from_current, global_from_car,car_from_global,ref_from_car]
                    ))
                    #print(trans_matrix)
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                agent_id=int(cam_type.split('_')[-1])
                #print(info['cams'].items(),len(trans_matrix))
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                #print(cam_type,lidar2img_rt,agent_id,trans_matrix[agent_id-1]) 
                #print(lidar2img_rt,trans_matrix[agent_id-1])
                lidar2img_rt=lidar2img_rt @ trans_matrix[agent_id-1]#7.19 add by hsz 不确定对不对
                #print(cam_type,lidar2img_rt)
                lidar2img_rts.append(lidar2img_rt)
                
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'][0])
        translation = input_dict['ego2global_translation'][0]
        can_bus = input_dict['can_bus'][0]
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            #print(self.prepare_test_data(idx))
            #pdb.set_trace()
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        eval_set_map = {
            'v2.0-mini': 'mini_val',
            'v1.0-mini': 'val',
            'v1.0-test': 'test',
        }
        from nuscenes import NuScenes
        if self.version!='v2.0-mini':
            
            self.nusc = NuScenes(version='v1.0-mini', dataroot=self.data_root,
                             verbose=True)
        else :
            
            self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)
        output_dir = osp.join(*osp.split(result_path)[:-1])
        
        
        self.nusc_eval = V2XSIMEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_V2X_SIM'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
