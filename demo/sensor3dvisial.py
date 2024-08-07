import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 常量定义
HORIZONTAL_SECTIONS = 180
VERTICAL_SECTIONS = 90
PI = np.pi

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
        distance = np.sqrt(x**2 + y**2 + z**2)

        # 计算方位角和俯仰角
        azimuth = np.arctan2(x, z)  # 使用 x 和 z 计算方位角
        elevation = np.arctan2(-y, np.sqrt(x**2 + z**2))  # 使用 -y 和 sqrt(x^2 + z^2) 计算俯仰角

        # 将方位角和俯仰角转换为球形网格索引
        horizontal_index = int((azimuth + PI) / (2 * PI) * HORIZONTAL_SECTIONS) % HORIZONTAL_SECTIONS
        vertical_index = int((elevation + PI/2) / PI * VERTICAL_SECTIONS) % VERTICAL_SECTIONS

        # 保留每个网格中距原点最近的点
        if distance < distance_matrix[vertical_index, horizontal_index]:
            distance_matrix[vertical_index, horizontal_index] = distance

    # 对矩阵里的数值进行归一化处理
    for v in range(VERTICAL_SECTIONS):
        for h in range(HORIZONTAL_SECTIONS):
            distance = distance_matrix[v, h]
            # 归一化处理并且距离大于0.5时设为0
            distance_matrix[v, h] = max(0.0, 1 - distance / 2) if distance != np.inf else 0.0

    return distance_matrix

def update_point_cloud():
    # 模拟生成新的点云数据
    # 你可以在这里插入实际的数据接收逻辑
    return np.random.uniform(-3, 3, (1000, 3))

def update_plt(self, ax, azimuths_grid, elevations_grid):
    ax.clear()

    # 获取新的点云数据
    point_cloud = update_point_cloud()

    # 计算距离矩阵
    distance_matrix = compute_distance_matrix(point_cloud)

    # 将球面坐标转换为笛卡尔坐标
    x = np.cos(elevations_grid) * np.sin(azimuths_grid)
    y = np.sin(elevations_grid)
    z = np.cos(elevations_grid) * np.cos(azimuths_grid)

    # 绘制球面图，设置alpha为1.0使球面透明
    ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(distance_matrix), rstride=1, cstride=1, alpha=1.0, edgecolor='none')

    # 设置轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # 设置标题
    ax.set_title('Real-Time Point Cloud Perception on Transparent Sphere')

def main():
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建角度网格
    azimuths = np.linspace(-PI, PI, HORIZONTAL_SECTIONS)
    elevations = np.linspace(-PI/2, PI/2, VERTICAL_SECTIONS)
    azimuths_grid, elevations_grid = np.meshgrid(azimuths, elevations)

    # 创建动画
    ani = FuncAnimation(fig, update_plt, fargs=(ax, azimuths_grid, elevations_grid), interval=100)

    # 显示图形
    plt.show()

if __name__ == '__main__':
    main()
