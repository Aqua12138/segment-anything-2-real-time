import numpy as np
import open3d as o3d
import time

# 常量定义
HORIZONTAL_SECTIONS = 180  # 方位角分区数
VERTICAL_SECTIONS = 90  # 俯仰角分区数
PI = np.pi
point_cloud = o3d.geometry.PointCloud()
sphere = o3d.visualization.Visualizer()
sphere.create_window('Masked Point Cloud')

def compute_distance_matrix(point_cloud, origin=(0.0, 0.0, 0.0)):
    # 初始化距离矩阵，填充为极大值
    distance_matrix = np.full((VERTICAL_SECTIONS, HORIZONTAL_SECTIONS), np.inf)

    # 获取原点坐标
    x0, y0, z0 = origin

    for point in point_cloud:
        # 计算点到原点的相对位置
        x = point[0] - x0
        y = point[1] - y0
        z = point[2] - z0
        distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # 计算方位角和俯仰角
        azimuth = np.arctan2(z, x)  # 使用 z 和 x 计算方位角
        elevation = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))  # 使用 y 和 sqrt(x^2 + z^2) 计算俯仰角

        # 将方位角和俯仰角转换为球形网格索引
        horizontal_index = int((azimuth + PI) / (2 * PI) * HORIZONTAL_SECTIONS) % HORIZONTAL_SECTIONS
        vertical_index = int((elevation + PI / 2) / PI * VERTICAL_SECTIONS) % VERTICAL_SECTIONS

        # 保留每个网格中距原点最近的点
        if distance < distance_matrix[vertical_index, horizontal_index]:
            distance_matrix[vertical_index, horizontal_index] = distance

    # 对矩阵里的数值进行归一化处理
    for v in range(VERTICAL_SECTIONS):
        for h in range(HORIZONTAL_SECTIONS):
            distance = distance_matrix[v, h]
            # 归一化处理并且距离大于0.5时设为0
            distance_matrix[v, h] = max(0.0, 1 - distance / 0.5) if distance != np.inf else 0.0

    return distance_matrix


def visualize_distance_matrix_on_sphere(distance_matrix):
    # 将距离矩阵投影到球面上
    points = []
    colors = []
    for v in range(VERTICAL_SECTIONS):
        for h in range(HORIZONTAL_SECTIONS):
            distance = distance_matrix[v, h]
            if distance > 0:
                # 将方位角和俯仰角转换为球面坐标（半径为1）
                azimuth = (h / HORIZONTAL_SECTIONS) * 2 * PI - PI
                elevation = (v / VERTICAL_SECTIONS) * PI - PI / 2
                x = np.cos(elevation) * np.cos(azimuth)
                y = np.sin(elevation)
                z = np.cos(elevation) * np.sin(azimuth)
                points.append([x, y, z])

                # 将距离转换为颜色
                color = distance
                colors.append([color, 1 - color, 0.5])

    # 创建Open3D点云对象

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    #o3d.visualization.draw_geometries([point_cloud])
    sphere.add_geometry(point_cloud)
    sphere.update_geometry(point_cloud)
    sphere.poll_events()
    sphere.update_renderer()

def get_new_point_cloud():
    # 模拟获取新的点云数据
    # 这里生成一些随机点用于演示
    num_points = 1000
    x = np.random.uniform(-2, 2, num_points)
    y = np.random.uniform(-2, 2, num_points)
    z = np.random.uniform(0, 2, num_points)  # 保证 z 不为 0
    return np.vstack((x, y, z)).T


def main():
    while True:
        # 获取新的点云数据
        point_cloud = get_new_point_cloud()

        # 计算距离矩阵
        distance_matrix = compute_distance_matrix(point_cloud)

        # 可视化距离矩阵
        visualize_distance_matrix_on_sphere(distance_matrix)

        # 模拟更新速度
        time.sleep(0.5)  # 延迟0.5秒以降低更新频率

        # 用户可以通过关闭Open3D的可视化窗口来停止程序


if __name__ == "__main__":
    main()
