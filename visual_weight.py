import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

#nusc_map = NuScenesMap(dataroot='/data/V2X-Sim-2.0', map_name='map')
  
def visual_weights(weights,bev_mask, index, trans_matrix,car,frame):
    
    # 创建一个（50x50）的随机图像，每个像素值在0到1之间
    image_size = (50, 50)
    #image = np.random.rand(*image_size)

    # 创建一个包含子图的画布
    fig = plt.figure(figsize=(12, 5))

    # 在第一个子图中绘制平面图像
    ax1 = fig.add_subplot(121)
    
    
    #ax1.set_title(cam_name)
    
    ax1.axis('off')
    
    # 定义一个示例的 4x4 转移矩阵
    '''trans_matrix = np.array([[2, 0, 0, 10],
                            [0, 1, 0, 5],
                            [0, 0, 1.5, -3],
                            [0, 0, 0, 1]])'''
    # 生成网格的坐标
    x_range = np.linspace(-51.2, 51.2, 50)
    y_range = np.linspace(-51.2, 51.2, 50)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    # 将平面坐标点变换到三维空间
    coords_plane = np.vstack((x_grid.flatten(), y_grid.flatten(), np.zeros(x_grid.size), np.ones(x_grid.size)))
    #new_coords_plane = np.dot(trans_matrix, coords_plane).reshape((4, 50, 50))
    image=np.zeros((int(np.sqrt(weights.shape[1])),int(np.sqrt(weights.shape[1]))))
    for cam in range(6):
        
        index_query_per_img=bev_mask[cam][0].sum(-1).nonzero()[0]
        weights_reshape=weights.reshape(weights.shape[0],int(np.sqrt(weights.shape[1])),int(np.sqrt(weights.shape[1])))
        #index_query_per_img_reshape=index_query_per_img.reshape(int(np.sqrt(index_query_per_img.shape[0])),int(np.sqrt(index_query_per_img.shape[0])))
        index_mask=np.zeros_like(weights[0])
        index_mask[index_query_per_img]=1
        #print(weights.shape)
        index_mask_reshape=index_mask.reshape(int(np.sqrt(weights.shape[1])),int(np.sqrt(weights.shape[1])))
        rotate_image=np.rot90(weights_reshape[cam]*index_mask_reshape)
        print(cam,(weights_reshape[cam]*index_mask_reshape).sum()/index_mask_reshape.sum())
        mirrored_image = np.flip(rotate_image, axis=1)
        image+=mirrored_image
    ax1.imshow(image, cmap='Reds', extent=(-51.2, 51.2, -51.2, 51.2), vmin=0, vmax=1)
        
    '''for i in range(index_query_per_img.shape[0]):
            proj_x=-(index_query_per_img[i]//50 -25)/25*  51.2
            proj_y = (index_query_per_img[i]%50-25)/25 * 51.2
            #ax.plot(proj_x, proj_y, 'ro', markersize=0.5)
            #weight=weights[i]
            #gray_color = (int(weight),  int(weight),  int(weight))
            #print(weight)
            color = plt.cm.gray(1)
            color = (0, 0.5, 0)
            #color = plt.cm.viridis(weight)  # 使用viridis颜色映射根据权重调整颜色
            ax1.plot(proj_x, proj_y, 'o', markersize=0.5, color=color)'''
    # 在第二个子图中绘制转移后的点
    #ax2 = fig.add_subplot(122)
    #ax2.imshow(image, cmap='gray', vmin=0, vmax=1)
    #ax2.scatter(new_coords_plane[0], new_coords_plane[1], color='red', marker='o', )
    #ax2.set_xlim(-51.2, 51.2)  # 设置 x 轴范围
    #ax2.set_ylim(-51.2, 51.2)  # 设置 y 轴范围
    
    def draw_car(ax, position, direction, color):
        car = FancyArrowPatch(
            position, direction, arrowstyle='->', mutation_scale=15, color=color
        )
        ax.add_patch(car)
    car_position = (0, 0)
    car_direction = (0, 1)
    draw_car(ax1, car_position, car_direction, 'blue')
    #ax1.scatter(0,0, color='blue', marker='x',) 
    #pdb.set_trace()
    x1=(trans_matrix[:3,:3].T @ trans_matrix[:3,3])[0]#19
    y1=(trans_matrix[:3,:3].T @ trans_matrix[:3,3])[1]#29
    car_position = (y1, -x1)
    print(trans_matrix)
    car_direction = (car_position[0]+trans_matrix[1,0],car_position[1]+trans_matrix[1,1])
    draw_car(ax1, car_position, car_direction, 'red')
    #ax1.scatter(-y1,x1, color='red', marker='x',) #label='Point (x1, y1)'  # 标注另一个点
    #ax1.axhline(y=0, color='yellow', linestyle='--', linewidth=0.8, label='X Axis')  # 绘制 y 坐标轴
    #ax1.axvline(x=0, color='orange', linestyle='--', linewidth=0.8, label='Y Axis')  # 绘制 x 坐标轴
    
    #ax2.set_title('weights with cars')
    #ax1.legend()
    #ax2.axis('off')

    plt.tight_layout()

    # 保存为图片
    plt.savefig('weights/combined_visualization_{}_{}.png'.format(frame,car), bbox_inches='tight', pad_inches=0.1)

    plt.close()





def projection(image_path,reference_points ,weights,index,cam_name,frame ):
    # 读取图像数据
    #image_path = 'your_image_path.jpg'  # 替换为您的图像路径
    image = Image.open(image_path)
    image = np.array(image) / 255.0  # 将像素值范围映射到 [0, 1]

    # 创建示例的参考点投影数据，假设有两个参考点，每个参考点有四层投影
    num_points = reference_points.shape[0]
    #reference_points = np.random.rand(num_points, 4, 2)  # [num_points, 4, 2]，每个点4层投影到2D图片

    # 创建包含子图的画布
    fig, ax = plt.subplots(figsize=(8, 8))
     #= np.random.rand(num_points, 4)
# 在子图中绘制图像
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)

    # 在图像上绘制所有参考点的投影
    for i in range(num_points):
        for layer in range(4):
            #应该是x长y短
            #pdb.set_trace()
            if 0<reference_points[i, layer,0]<1 and 0<reference_points[i, layer,1]<1:
                proj_x = reference_points[i, layer, 0] * image.shape[1]
                proj_y = (reference_points[i, layer, 1]) * image.shape[0]
                #ax.plot(proj_x, proj_y, 'ro', markersize=0.5)
                weight=weights[i]
                weight=1# for fully transparency
                #gray_color = (int(weight),  int(weight),  int(weight))
                #print(weight)
                color = plt.cm.gray(1)
                color = (0, 0, 0, 1-float(weight))
                #color = plt.cm.viridis(weight)  # 使用viridis颜色映射根据权重调整颜色
                ax.plot(proj_x, proj_y, 'o', markersize=0.5, color=color)

    # 设置图像属性
    ax.set_title(cam_name)
    ax.axis('off')
    #ax.legend()

    plt.tight_layout()

    # 保存为图片
    plt.savefig('weights/projected_reference_points_visualization_{}_{}.png'.format(frame,index), bbox_inches='tight', pad_inches=0.1)
    #print("投影后的参考点图像已保存为 projected_reference_points_visualization.png")
    plt.close()

if __name__=='__main__':
    #os.mkdir('weights')
    #frame=4
    for frame in range(101,104):
        weights=np.load("weight{}.npy".format(frame))
        bev_mask=np.load("bev_mask{}.npy".format(frame))
        
        kwargs=np.load('kwargs{}.npy'.format(frame),allow_pickle=True)
        img_metas=dict(enumerate(kwargs.flatten(), 1))[1]['img_metas'][0]
        #['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor', 'pts_filename', 'scene_token', 'can_bus'])
        reference_pts=np.load('reference_pts{}.npy'.format(frame))
        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        weights_reshape=weights.reshape(weights.shape[0],int(np.sqrt(weights.shape[1])),int(np.sqrt(weights.shape[1])))
        #pdb.set_trace()
        lidar2img=img_metas['lidar2img']
        trans_matrix=np.linalg.inv(lidar2img[0])@lidar2img[6]
        x1=(-trans_matrix[:3,:3].T @ trans_matrix[:3,3])[0]
        y1=(-trans_matrix[:3,:3].T @ trans_matrix[:3,3])[1]
        for car in range(weights.shape[0]//6):
            visual_weights(weights[6*car:6+6*car],bev_mask[6*car:6+6*car],None,trans_matrix,car,frame)
        for cam in range(weights.shape[0]): 
            #pdb.set_trace()
            image_path=img_metas['filename'][cam]
            cam_name=image_path.split('/')[-2]
            
            
            pdb.set_trace()
            
            index_query_per_img=bev_mask[cam][0].sum(-1).nonzero()[0]
            #ref_3d=np.linalg.inv(lidar2img[cam])@reference_pts[cam][0][index_query_per_img]
            #print(x1,y1)
            
            #
            
            
            print(index_query_per_img.shape)
            
            
            #pdb.set_trace()
            
            projection(image_path,reference_pts[cam][0][index_query_per_img],weights[cam][index_query_per_img],cam,cam_name, frame)