import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import taichi as ti
import time
from scipy.spatial.transform import Rotation as R
from my_online import SamOnline
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

ti.init(arch=ti.cuda, device_memory_GB=5)
@ti.data_oriented
class seg3d:
    def __init__(self, RES_X, RES_Y, headless):
        #姿态
        self.euler_angles = None

        self.init_camera(RES_X, RES_Y)
        self.sam2 = SamOnline("/home/zhx/Project/segment-anything-2/checkpoints/sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml", headless=True)
        self.rgb = None
        self.verts = None
        self.headless = headless

        # 创建Open3D点云对象和可视化器
        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud")
        self.view_ctl = self.vis.get_view_control()

        # 创建一次Ti数组，后续循环中复用
        self.max_point = RES_X * RES_Y
        self.ti_verts = ti.Vector.field(3, dtype=ti.f32, shape=self.max_point)  # 假定点云最多307200点
        self.ti_mask = ti.field(dtype=ti.i32, shape=(RES_X, RES_Y))  # 对应摄像机分辨率
        self.filtered_indices = ti.field(dtype=ti.i32, shape=self.max_point)
        self.RES_X = RES_X
        self.RES_Y = RES_Y

        #网格
        self.GRID_SIZE = 0.05  # 每个格子的尺寸，这里是5cm
        self.GRID_COUNT = 40
        #3D sensor
        self.HORIZONTAL_SECTIONS = 180
        self.VERTICAL_SECTIONS = 90
        self.PI = np.pi


    @ti.kernel
    def filter_points(self):
        for i in range(self.max_point):
            vec = self.ti_verts[i]
            x, y, z = vec.x, vec.y, vec.z
            if z > 0 and z < 2:
                pixel_x = (x / z) * self.intrinsics_fx + self.intrinsics_ppx
                pixel_y = (y / z) * self.intrinsics_fy + self.intrinsics_ppy
                if pixel_x > 0 and pixel_x < self.RES_X - 1 and pixel_y > 0 and pixel_y < self.RES_Y - 1:
                    if self.ti_mask[int(pixel_x), int(pixel_y)] != 0:
                        self.filtered_indices[i] = 1

    def init_camera(self, RES_X, RES_Y):
        # 初始化相机
        self.pipeline = rs.pipeline()
        config = rs.config()
        #包含加速度计和陀螺仪
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 100)
        config.enable_stream(rs.stream.color, RES_X, RES_Y, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.holes_fill, 3)

        #对齐深度和rgb
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.color_frame = aligned_frames.get_color_frame()

        # 获取相机内参
        intrinsics = self.color_frame.profile.as_video_stream_profile().intrinsics
        self.intrinsics_fx = intrinsics.fx
        self.intrinsics_fy = intrinsics.fy
        self.intrinsics_ppx = intrinsics.ppx
        self.intrinsics_ppy = intrinsics.ppy

        #获取姿态数据
        rotation = R.from_euler('xyz', [0, 0, 0], degrees=True) #初始化
        accel = aligned_frames.first_or_default(rs.stream.accel)
        gyro = aligned_frames.first_or_default(rs.stream.gyro)
        if accel and gyro:
            accel_data = accel.as_motion_frame().get_motion_data()
            gyro_data = gyro.as_motion_frame().get_motion_data()
            # 将陀螺仪数据转换为旋转向量（单位是弧度/秒）
            gyro_vector = np.array([gyro_data.x, gyro_data.y, gyro_data.z])
            dt = 1.0 / 200  # 采样间隔，假设200Hz采样率
            delta_rotation = R.from_rotvec(gyro_vector * dt)
            # 更新姿态
            rotation = delta_rotation * rotation
            # 获取当前姿态
            self.euler_angles = rotation.as_euler('xyz', degrees=True)

    @property
    def get_rgb(self):
        return self.rgb
    @property
    def get_verts(self):
        return self.verts

    def update_camera(self):
        # 等待一帧数据
        start = time.time()
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        self.color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = self.spatial.process(depth_frame)
        # 获取RGB图像
        self.rgb = np.asanyarray(self.color_frame.get_data())

        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        v = points.get_vertices()
        self.verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # x, y, z
        self.ti_verts.from_numpy(self.verts)

    def compute_mask(self):
        mask = self.sam2.seg_image(self.get_rgb)
        # 定义结构元素（核）
        kernel_size = 5  # 可以调整核的大小
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 腐蚀操作
        self.mask_np = cv2.erode(mask, kernel, iterations=2).astype(np.uint8).transpose((1, 0))

        self.ti_mask.fill(0)
        self.ti_mask.from_numpy(self.mask_np)

    def seg_pc(self):
        self.filtered_indices.fill(0)
        # 使用Taichi过滤点云
        self.filter_points()

        # 提取过滤后的点云
        self.filtered_points = self.verts[self.filtered_indices.to_numpy() == 1]

    def get_rotation_matrix(self):
        # Yaw rotation matrix (around z-axis)
        yaw = self.euler_angles[0]
        pitch = self.euler_angles[1]
        roll = self.euler_angles[2]
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Pitch rotation matrix (around y-axis)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Roll rotation matrix (around x-axis)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # Combined rotation matrix
        R = R_z @ R_y @ R_x
        return R

    def get_transformation_matrix(self):
        # Get the rotation matrix from yaw, pitch, roll
        R = self.get_rotation_matrix()

        # Define the coordinate transformation matrix
        T = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])

        # The total transformation matrix
        M = T @ R
        return M

    def transform_point_cloud(self):
        # Calculate the transformation matrix
        M = self.get_transformation_matrix()

        # Apply the transformation to each point in the point cloud
        transformed_points = np.dot(self.filtered_points, M.T)

        return transformed_points

    def point_cloud_2D(self):
        # 初始化网格矩阵，所有值初始为0
        grid = np.zeros((self.GRID_COUNT, self.GRID_COUNT), dtype=int)

        # 遍历点云中的每个点，将它们映射到网格中
        for point in self.transform_point_cloud():
            # 计算点在网格中的索引 注意水平面对应相机的x-z平面
            x_index = int(np.floor((point[0] + 1) / self.GRID_SIZE))
            y_index = int(np.floor((point[2]) / self.GRID_SIZE))

            # 如果索引在网格范围内，设置相应的格子为1
            if 0 <= abs(x_index) < self.GRID_COUNT and 0 <= abs(y_index) < self.GRID_COUNT:
                grid[y_index, x_index] = 1

        return grid

    def point_cloud_3D(self, origin=(0.0, 0.0, 1.0)):
        # 初始化距离矩阵，填充为极大值
        distance_matrix = np.full((self.VERTICAL_SECTIONS, self.HORIZONTAL_SECTIONS), np.inf)

        # 获取原点坐标
        x0, y0, z0 = origin

        for point in self.transform_point_cloud():
            # 计算点到原点的相对位置
            x = point[0] - x0
            y = point[1] - y0
            z = point[2] - z0
            distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

            # 计算方位角和俯仰角
            azimuth = np.arctan2(x, z)
            elevation = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))

            # 将方位角和俯仰角转换为球形网格索引
            horizontal_index = int((azimuth) / (2 * self.PI) * self.HORIZONTAL_SECTIONS) % self.HORIZONTAL_SECTIONS
            vertical_index = int((elevation + self.PI / 2) / self.PI * self.VERTICAL_SECTIONS) % self.VERTICAL_SECTIONS

            # 保留每个网格中距原点最近的点
            if distance < distance_matrix[vertical_index, horizontal_index]:
                distance_matrix[vertical_index, horizontal_index] = distance

        # 对矩阵里的数值进行归一化处理
        for v in range(self.VERTICAL_SECTIONS):
            for h in range(self.HORIZONTAL_SECTIONS):
                distance = distance_matrix[v, h]
                # 归一化处理并且距离大于1时设为0
                distance_matrix[v, h] = max(0.0, 1 - distance / 2) if distance != np.inf else 0.0

        return distance_matrix
    def visualize_grid_2D(self, grid):
        # 将二维网格矩阵转换为图像
        img = np.uint8(grid * 255)  # 将0/1矩阵转换为0/255
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)  # 放大图像以便可视化

        # 显示图像
        # cv2.imshow("Grid Visualization", img)
        # cv2.waitKey(1)  # 等待1毫秒以显示图像

    def visualize_grid_3D(self, distance_matrix):
        # 将二维网格矩阵转换为图像
        img = np.uint8(distance_matrix * 255)
        color_mapped_image = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)# 将0/1矩阵转换为0/255
        img = cv2.resize(img, (1000, 500), interpolation=cv2.INTER_NEAREST)  # 放大图像以便可视化

        # 显示图像
        # cv2.imshow("Distance Matrix Visualization", img)
        # cv2.waitKey(1)  # 等待1毫秒以显示图像

    def update_point_cloud(self):
        # 模拟生成新的点云数据
        # 你可以在这里插入实际的数据接收逻辑
        return np.random.uniform(-3, 3, (1000, 3))

    def update_plt(self, frame, ax, azimuths_grid, elevations_grid):
        ax.clear()
        #更新
        self.update()
        # 获取
        distance_matrix = self.point_cloud_3D()

        # 将球面坐标转换为笛卡尔坐标
        x = np.cos(elevations_grid) * np.sin(azimuths_grid)
        y = np.sin(elevations_grid)
        z = np.cos(elevations_grid) * np.cos(azimuths_grid)

        # 绘制球面图，设置alpha为1.0使球面透明
        ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(distance_matrix), rstride=1, cstride=1, alpha=0.3,
                        edgecolor='none')

        # 设置轴标签
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # 设置标题
        ax.set_title('Real-Time Point Cloud Perception on Transparent Sphere')

    def render(self):
        # Segmentation RGB
        colored_mask = np.zeros_like(self.rgb)
        colored_mask[self.mask_np.transpose((1, 0)) > 0] = [0, 255, 0]  # 绿色掩码
        image = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)
        combined_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        cv2.imshow('Segmentation RGB', combined_image)
        cv2.waitKey(1)

        # Segmentation Point Cloud
        print(f'Number of points in the polygon: {len(self.filtered_points)}')
        # self.point_cloud.points = o3d.utility.Vector3dVector(filtered_points.astype(np.float64))
        # pcd = o3d.geometry.PointCloud()
        self.pcd.clear()
        self.pcd.points = o3d.utility.Vector3dVector(-self.filtered_points.astype(np.float64))
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([self.pcd])
        # 更新可视化器
        self.vis.add_geometry(self.pcd)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def update(self):
        time1 = time.time()
        self.update_camera()
        time2 = time.time()
        print('update_camera:', time2-time1)
        self.compute_mask()
        time3 = time.time()
        print('compute_mask', time3-time2)
        self.seg_pc()
        time4 = time.time()
        print('seg_pc', time4-time3)
        if not self.headless:
            self.render()

if __name__ == '__main__':
    Seg3D = seg3d(960, 540, False)
    # next
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建角度网格
    azimuths = np.linspace(-np.pi, np.pi, Seg3D.HORIZONTAL_SECTIONS)
    elevations = np.linspace(-np.pi / 2, np.pi / 2, Seg3D.VERTICAL_SECTIONS)
    azimuths_grid, elevations_grid = np.meshgrid(azimuths, elevations)

    # 创建动画
    ani = FuncAnimation(fig, Seg3D.update_plt, fargs=(ax, azimuths_grid, elevations_grid), interval=100)

    # 显示图形
    plt.show()






