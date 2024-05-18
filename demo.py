import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from yacs.config import CfgNode as CN
from config import cfg
import cv2
import numpy as np
from utils.transforms import get_affine_transform
from core.inference import  get_final_preds
from models.pose_hrnet import get_pose_net
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEBUG = True
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

MPII_KEYPOINT_INDEXES = {
    0: "right ankle",
    1: "right knee",
    2: "right hip", 
    3: "left hip", 
    4: "left knee", 
    5: "left ankle",
    6: "pelvis", 
    7: "thorax", 
    8: "upper neck", 
    9: "head top", 
    10: "right wrist",
    11: "right elbow", 
    12: "right shoulder", 
    13: "left shoulder", 
    14: "left elbow",
    15: "left wrist"
}

SKELETON = [
    [0, 1], [1, 2], [2, 6], 
    [5, 4], [4, 3], [3, 6],
    [6, 7], [7, 8], [8, 9],
    [10, 11], [11, 12], [12, 7],
    [15, 14], [14, 13], [13, 7]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = len(MPII_KEYPOINT_INDEXES.keys())

def draw_pose(keypoints, img, joint_thickness=6):
    assert keypoints.shape == (NUM_KPTS, 2)
    for i, (kpt_a, kpt_b) in enumerate(SKELETON):
        x_a, y_a = keypoints[kpt_a]
        x_b, y_b = keypoints[kpt_b]
        cv2.circle(img, (int(x_a), int(y_a)), joint_thickness, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), joint_thickness, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def calculate_center_scale(box):
  x1, y1, x2, y2 = box
  center_x = (x2 + x1) / 2
  center_y = (y2 + y1) / 2
  width = x2 - x1
  height = y2 - y1
  
  center = np.array([center_x, center_y], dtype=np.float32)
  scale = max(width, height) / 200

  return center, scale

def get_person_box(model, img):
  results = model(img)
  
  data = results.xyxy[0].cpu().numpy()
    
  if len(data) == 0:
    return np.array([-1])

  max_index = np.argmax(data[:, 4])
  max_element = data[max_index]

  center, scale = calculate_center_scale(max_element[:4])

  return center, scale
    
def load_yolo(version):
  yolo = torch.hub.load('ultralytics/yolov5', version, 
                        pretrained= True, _verbose= False)
  yolo.cuda()
  yolo.classes = [0]
  return yolo

def load_hrnet(cfg):
  pose_model = get_pose_net(cfg, is_train=False)
    
  print(cfg.TEST.MODEL_FILE)
  if cfg.TEST.MODEL_FILE:
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
  else:
    print('expected model defined in config at TEST.MODEL_FILE')

  pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
  pose_model.to(CTX)
  pose_model.eval()

  return pose_model
if __name__ == '__main__':
    pose_model = load_hrnet(cfg)
    box_model = load_yolo("yolov5s")
    image_bgr = cv2.imread("image-mpii/image/000550580.jpg")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    center, scale = get_person_box(box_model, image)
    image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
    pose_pred = get_pose_estimation_prediction(pose_model, image_pose, center, scale)[0]
    draw_pose(pose_pred,image_bgr)
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, image_bgr)
    print(f'The result image has been saved as {output_path}')