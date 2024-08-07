import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import taichi as ti
import time
from scipy.spatial.transform import Rotation as R
from my_online import SamOnline

ti.init(arch=ti.cuda, device_memory_GB=5)
@ti.data_oriented
class seg3d:
    def __init__(self, RES_X, RES_Y, headless):
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

        self.camera_base = np.array([
            [0, 0, 1, -3.6565167903900146],
            [0, -1, 0, 1.554892659187317],
            [1, 0, 0, 7.2938642501831055],
            [0, 0, 0, 1.0]
        ])

        self.pixel_axis = np.array([
            [1, 0, 0, -3.1656227111816406],
            [0, 1, 0, 1.554892659187317],
            [0, 0, 1, 7.219107627868652],
            [0, 0, 0, 1.0]
        ])
        self.init_camera(RES_X, RES_Y)

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
        accel = aligned_frames.first_or_default(rs.stream.accel)
        if accel:
            accel_data = accel.as_motion_frame().get_motion_data()
            theta = np.arctan2(-accel_data.z, accel_data.y) - np.pi
            Camera_Rx = np.array([
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]
            ])

            self.camera_axis = self.camera_base @ Camera_Rx

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

    def get_transformation_matrix(self):
        M = np.linalg.inv(self.pixel_axis).dot(self.camera_axis)
        return M


    def transform_point_cloud(self):
        # 计算变换矩阵
        M = self.get_transformation_matrix()

        # 将点云扩展到齐次坐标空间
        ones = np.ones((self.filtered_points.shape[0], 1))
        homogeneous_points = np.hstack((self.filtered_points, ones))

        # 应用变换矩阵到每个点上
        transformed_points = np.dot(M, homogeneous_points.T)
        # 返回变换后的点云，不包括齐次坐标
        return transformed_points[:3, :]

    def process_point_cloud(self):
        # 初始化网格矩阵，所有值初始为0
        grid = np.zeros((self.GRID_COUNT, self.GRID_COUNT), dtype=int)
        # 遍历点云中的每个点，将它们映射到网格中
        for point in self.transform_point_cloud().T:
            # 计算点在网格中的索引 注意水平面对应相机的x-z平面
            x_index = int(np.floor((point[0]) / self.GRID_SIZE))
            z_index = int(np.floor((point[2]) / self.GRID_SIZE))

            # 如果索引在网格范围内，设置相应的格子为1
            if -1/2 * self.GRID_COUNT <= x_index <= (1/2 * self.GRID_COUNT) - 1 and -1/2 * self.GRID_COUNT <= z_index <= (1/2 * self.GRID_COUNT) - 1:
                grid[z_index + int(1/2 * self.GRID_COUNT), x_index + int(1/2 * self.GRID_COUNT)] = 1
        return grid

    def visualize_grid(self, grid):
        # 将二维网格矩阵转换为图像
        img = np.uint8(grid * 255)  # 将0/1矩阵转换为0/255
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)  # 放大图像以便可视化

        # 显示图像
        cv2.imshow("Grid Visualization", img)
        cv2.waitKey(1)  # 等待1毫秒以显示图像

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
    while True:
        start = time.time()
        Seg3D.update()
        grid=Seg3D.process_point_cloud()
        Seg3D.visualize_grid(grid)
        end = time.time()






