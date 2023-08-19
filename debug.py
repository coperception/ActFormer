import mmcv
import numpy as np
import os
import glob
from nuscenes.nuscenes import NuScenes as V2XSimDataset

if __name__ == '__main__':
    v2x_sim=V2XSimDataset(version='v1.0-mini', dataroot='./data/V2X-Sim-2.0', verbose=True)
    for anno in v2x_sim.sample_annotation:
        print(anno['num_lidar_pts'])
            
    