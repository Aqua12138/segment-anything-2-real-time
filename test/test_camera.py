import pyrealsense2 as rs

# 创建管道
pipeline = rs.pipeline()

# 创建配置并配置请求深度流
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 开始流
pipeline.start(config)

try:
    # 等待一帧深度图
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth:
        raise RuntimeError("无法获取深度帧；有可能是相机未连接。")

    # 打印一些深度信息在中心点
    width, height = depth.get_width(), depth.get_height()
    center_depth = depth.get_distance(width // 2, height // 2)
    print(f"中心点深度: {center_depth:.2f} 米")

finally:
    # 停止流
    pipeline.stop()
