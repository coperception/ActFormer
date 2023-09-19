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
import re
import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

#nusc_map = NuScenesMap(dataroot='/data/V2X-Sim-2.0', map_name='map')
#origin [428, 508, 508, 742, 476, 476, 882, 915, 400, 209, 208, 483, 898, 854, 462, 209, 239, 447]

#new [428, 508, 508, 742, 476, 476, 548, 360, 344, 208, 208, 443, 531, 372, 399, 208, 239, 413]

def effeciency(file_path ):
    # 打开 .out 文件并读取内容
    #file_path = "your_file_path.out"  # 替换为实际的文件路径
    with open(file_path, "r",encoding='latin-1') as f:
        #print(f.readline())
        lines = f.readlines()
    #print(lines[:40])
    # 定义特定单词
    target_word1 = "origin"  # 替换为您的特定单词
    target_word2 = "new ["  
    averages=[]
    averages1 = []
    averages2 = []
    averages_non_ego=[]
    averages2_non_ego=[]
    averages_ego=[]
    averages2_ego=[]
    current_values = []
    current_values1 = []
    current_values2 = []
    for line in lines:
        
        if target_word1 in line:
            # 提取列表
            pattern = r"\[([\d,\s]+)\]"
            match = re.search(pattern, line)

            if match:
                # 提取匹配到的列表字符串
                list_str = match.group(1)
                
                # 将字符串转换为整数列表
                values_list = [int(num) for num in list_str.split(",")]
            #
            #print(line)
            current_values.extend(values_list)
        if current_values:
            # 计算当前列表的平均值
            #pdb.set_trace()
            average = max(current_values)*len(current_values)
            average_non_ego=max(current_values)*(len(current_values)-6)
            #print('origin',len(current_values))
            averages.append(average)
            averages_non_ego.append(average_non_ego)
            averages_ego.append( max(current_values)*6)
            current_values = []
        if target_word2 in line:
            # 提取列表
            pattern = r"\[([\d,\s]+)\]"
            match1 = re.search(pattern, line)
            #print(match1)
            if match1:
                # 提取匹配到的列表字符串
                list_str1 = match1.group(1)
                #print([int(num) for num in list_str1.split(",")])
                #pdb.set_trace()
                # 将字符串转换为整数列表
                values_list1 = [int(num) for num in list_str1.split(",")]
            #
            #print(line)
            current_values1.extend(values_list1)
            #print(sum(current_values1))
        if current_values1:
            # 计算当前列表的平均值
            #pdb.set_trace()
            #print(current_values1)
            average1 = max(current_values1[:6]) * 6
            if len(current_values1)>6:
                #print(len(current_values1))
                average2 = max(current_values1[6:])* (len(current_values1)-6)
            else: 
                average2=0
            averages1.append(average1+average2)
            averages2.append( max(current_values1)*len(current_values1))
            #print('new',len(current_values1))
            if len(current_values1)>6:
                averages2_non_ego.append( max(current_values1[6:])*(len(current_values1)-6))
            else: 
                averages2_non_ego.append(0)
            averages2_ego.append( max(current_values1[:6])*6)
            current_values1 = []
    #pdb.set_trace()
    # 输出每个列表的平均值
    print('Ours new efficiency:',sum(averages1)/sum(averages))
    print('average query:',sum(averages1)/ len(averages1),sum(averages)/ len(averages))
    print('Ours old efficiency:',sum(averages2)/sum(averages))
    #print('Ours non_ego:',sum(averages2_non_ego)/sum(averages_non_ego))
    print('Ours ego:',sum(averages2_ego)/sum(averages_ego))
    #print(averages1)
   
def draw_hist(list1,list2): 
    
    # 绘制第一个列表的直方图
    x_positions = np.arange(1,len(list1)+1)
    bar_width=0.4
# 绘制线性折线图
    plt.bar(x_positions, list1, width=bar_width, color='blue', label='BEVFormer')
    plt.bar(x_positions + bar_width, list2, width=bar_width, color='orange', label='Ours')

    # 设置 x 轴标签和标题
    plt.xlabel('Num of cars')
    plt.ylabel('Queries usage')
    plt.title('Comparison of BEVFormer and Ours')
    plt.xticks(x_positions, x_positions)  # 设置 x 轴刻度
    plt.legend()

    # 保存图像
    plt.savefig('comparison_plot.png', bbox_inches='tight')
    plt.close()

    x_positions = np.arange(1,7)
    #print(len(list1[6:]),len(x_positions ))
    bar_width=0.4
# 绘制线性折线图
    plt.bar(x_positions, list1[12:], width=bar_width, color='blue', label='BEVFormer')
    plt.bar(x_positions + bar_width, list2[12:], width=bar_width, color='orange', label='Ours')

    # 设置 x 轴标签和标题
    plt.xlabel('Camera index')
    plt.ylabel('Queries used')
    plt.title('Comparison of fully query and ours')
    plt.xticks(x_positions, x_positions)  # 设置 x 轴刻度
    plt.legend()

    # 保存图像
    plt.savefig('comparison_plot_non_ego.png', bbox_inches='tight')
    plt.close()

def draw_performance(list1,list2):
    #numbers = [10, 25, 15, 30, 20]

# 创建 x 轴的位置数组
    x_positions =range(1,len(list1)+1)
    plt.xticks(x_positions, x_positions)
    # 绘制线形图
    plt.plot(x_positions, list1, marker='o',label='Before')
    
    plt.plot(x_positions, list2, marker='s', label='After')

    # 设置 x 轴标签和标题
    plt.xlabel('Num of cars')
    plt.ylabel('AP')
    plt.title('Performance')
    
    plt.legend()
    plt.savefig('performance.png', bbox_inches='tight')
    plt.close()
    
def draw_percentage(list1):
    #numbers = [10, 25, 15, 30, 20]

# 创建 x 轴的位置数组
    x_positions =range(2,len(list1)+2)
    plt.xticks(x_positions, x_positions)
    # 绘制线形图
    plt.plot(x_positions, list1, marker='o',label='Number of queries percentage')
    
    
    # 设置 x 轴标签和标题
    plt.xlabel('Num of cars')
    plt.ylabel('Percentage')
    plt.title('Efficiency')
    
    plt.legend()
    plt.savefig('efficiency.png', bbox_inches='tight')
    plt.close()

def visual_weights(weights,bev_mask, index, lidar2img,car,frame):
    
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
    ax1.imshow(image, cmap='cividis', extent=(-51.2, 51.2, -51.2, 51.2), vmin=0, vmax=1)
        
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
        
        
        '''from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        import matplotlib.patches as patches
        car_body = patches.Rectangle((0, 0), 1, 0.4, linewidth=1, edgecolor='black', facecolor='blue')
        ax.add_patch(car_body)'''
    car_position = (0, 0)
    car_direction = (0, 1)
    draw_car(ax1, car_position, car_direction, 'blue')
    #ax1.scatter(0,0, color='blue', marker='x',) 
    #pdb.set_trace()
    for i in range(len(lidar2img)//6-1):
        trans_matrix=np.linalg.inv(lidar2img[0])@lidar2img[6+6*i]
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


    #LidarPointCloud.from_file(osp.join(nusc.dataroot, current_sd_rec["filename"]))
    #scene_86_000005
    
    #%4179 3591&4367,&4759,4284
if __name__=='__main__':
    #os.mkdir('weights')
    #frame=4
    
    performance1=[52.1,53.8,55.0,56.1]
    performance2=[51.7,54.8,55.8,58.9]
    percentage=[0.5328295901660738,0.6501549928628424,0.505291334707005]
    draw_performance(performance1,performance2)
    #draw_percentage(percentage)
    effeciency("/scratch/sh7748/test_5_cam_0.5.out")
    origin =[428, 508, 508, 742, 476, 476, 882, 915, 400, 209, 208, 483, 898, 854, 462, 209, 239, 447]

    new =[428, 508, 508, 742, 476, 476, 548, 360, 344, 208, 208, 443, 531, 372, 399, 208, 239, 413]

    draw_hist(origin,new)
    
    for frame in range(401,401):
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
        pdb.set_trace()
        for car in range(weights.shape[0]//6):
            visual_weights(weights[6*car:6+6*car],bev_mask[6*car:6+6*car],None,lidar2img,car,frame)
        for cam in range(weights.shape[0]): 
            #pdb.set_trace()
            image_path=img_metas['filename'][cam]
            cam_name=image_path.split('/')[-2]
            
            
            #pdb.set_trace()
            
            index_query_per_img=bev_mask[cam][0].sum(-1).nonzero()[0]
            #ref_3d=np.linalg.inv(lidar2img[cam])@reference_pts[cam][0][index_query_per_img]
            #print(x1,y1)
            
            #
            
            
            print(index_query_per_img.shape)
            
            
            #pdb.set_trace()
            
            projection(image_path,reference_pts[cam][0][index_query_per_img],weights[cam][index_query_per_img],cam,cam_name, frame)