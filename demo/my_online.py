import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_camera_predictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import time
import os
import cv2
import matplotlib.animation as animation
from natsort import natsorted
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
class SamOnline:
    def __init__(self, model, model_cfg, headless=True, prompts=None):
        sam2_checkpoint = model
        model_cfg = model_cfg
        self.headless = headless
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

        self.points = []
        self.labels = []

        self.if_init = False
        self.init_done = False  # 标志位，判断用户是否完成初始化

    def on_mouse_move(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse_position = (x, y)
    def seg_image(self, frame=None):
        if not self.if_init:
            self.width, self.height = frame.shape[:2][::-1]
            cv2.namedWindow('frame')
            cv2.setMouseCallback('frame', self.on_mouse_move)

            while True:
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 按下 'q' 键，记录标签 1
                    self.points.append(list(self.current_mouse_position))
                    self.labels.append(1)
                    print(f"Point added: {self.current_mouse_position} with label 1")
                elif key == ord('w'):  # 按下 'w' 键，记录标签 0
                    self.points.append(list(self.current_mouse_position))
                    self.labels.append(0)
                    print(f"Point added: {self.current_mouse_position} with label 0")
                elif key == 13:  # 按下回车键结束输入
                    break
            cv2.destroyAllWindows()

            self.predictor.load_first_frame(frame)
            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=self.points,
                labels=self.labels,
            )
            all_mask = np.zeros((self.height, self.width, 1), dtype=np.uint8)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                ) * 255
                all_mask = cv2.bitwise_or(all_mask, out_mask)
            self.if_init=True
        else:
            out_obj_ids, out_mask_logits = self.predictor.track(frame)

            all_mask = np.zeros((self.height, self.width, 1), dtype=np.uint8)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                ) * 255
                all_mask = cv2.bitwise_or(all_mask, out_mask)
        return all_mask

if __name__ == '__main__':
    directory_path = '/home/zhx/Project/segment-anything-2-real-time/video/frames'
    files = os.listdir(directory_path)
    files = natsorted(files)

    SO = SamOnline("../checkpoints/sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml", headless=False)
    while True:
        for file in files:
            frame = Image.open(os.path.join(directory_path, file))
            frame = np.array(frame.convert("RGB"))
            SO.seg_image(frame)







