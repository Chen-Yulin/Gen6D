import argparse
import subprocess
from pathlib import Path

import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
from skimage.io import imsave, imread
from tqdm import tqdm

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from eval import visualize_intermediate_results
from prepare import video2image
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp
import time
import UdpComms as U

def ObjectMsg(cat,px,py,pz,rx,ry,rz,sx,sy,sz):
    res = "{Object Detection}\n"
    res += cat + "\n"
    res += str(px) + "\n"
    res += str(py) + "\n"
    res += str(pz) + "\n"
    res += str(rx) + "\n"
    res += str(ry) + "\n"
    res += str(rz) + "\n"
    res += str(sx) + "\n"
    res += str(sy) + "\n"
    res += str(sz)
    return res

def weighted_pts(pts_list, weight_num=10, std_inv=10):
    weights=np.exp(-(np.arange(weight_num)/std_inv)**2)[::-1] # wn
    pose_num=len(pts_list)
    if pose_num<weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:,None,None],0)/np.sum(weights)
    return pts

def main(args):
    cfg = load_cfg(args.cfg)
    ref_database = parse_database_name(args.database)
    estimator = name2estimator[cfg['type']](cfg)
    estimator.build(ref_database, split_type='all')
    

    object_pts = get_ref_point_cloud(ref_database)
    object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

    #output_dir = Path(args.output)
    #output_dir.mkdir(exist_ok=True, parents=True)

    #(output_dir / 'images_raw').mkdir(exist_ok=True, parents=True)
    #(output_dir / 'images_out').mkdir(exist_ok=True, parents=True)
    #(output_dir / 'images_inter').mkdir(exist_ok=True, parents=True)
    #(output_dir / 'images_out_smooth').mkdir(exist_ok=True, parents=True)

    #que_num = video2image(args.video, output_dir/'images_raw', 1, args.resolution, args.transpose)
    cap = cv2.VideoCapture(0)
    pose_init = None
    hist_pts = []
    i = 0
    sock = U.UdpComms(udpIP="127.0.0.1", portTX=8002, portRX=8003, enableRX=True, suppressWarnings=True)
    cat = "Part1"
    pos = [-0.2,1.7,0.2]
    rot = [0,0,0]
    size = [0.095,0.084,0.039]
    while i<100:
        i=i+1
        ret, img = cap.read()
        time.sleep(0.01)
    while True:
        ret, img = cap.read()
        # generate a pseudo K
        h, w, _ = img.shape
        f=np.sqrt(h**2+w**2)
        K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)

        if pose_init is not None:
            estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization
        pose_pr, inter_results = estimator.predict(img, K, pose_init=pose_init)
        pose_init = pose_pr

        pts, _ = project_points(object_bbox_3d, pose_pr, K)
        #bbox_img = draw_bbox_3d(img, pts, (0,0,255))
        #imsave(f'{str(output_dir)}/images_out/{que_id}-bbox.jpg', bbox_img)
        #np.save(f'{str(output_dir)}/images_out/{que_id}-pose.npy', pose_pr)
        #cv2.imshow("Inter", visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d))
        bbox_inter = visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d)

        #print(bbox_inter.shape)
        bbox_inter = cv2.resize(bbox_inter, (640, 480))


        hist_pts.append(pts)
        pts_ = weighted_pts(hist_pts, weight_num=args.num, std_inv=args.std)
        pose_ = pnp(object_bbox_3d, pts_, K)
        #print(pose_)
        rotation_matrix = pose_[:, :3]
        translation_vector = pose_[:, 3]

        # 创建旋转对象
        rotation = R.from_matrix(np.transpose(rotation_matrix))

        # 将旋转矩阵转换为欧拉角
        euler_angles = rotation.as_euler('xyz', degrees=True)
        translation_vector = translation_vector*0.0481
        print(translation_vector,euler_angles)
        sock.SendData(ObjectMsg(cat, translation_vector[0], translation_vector[1], translation_vector[2], euler_angles[0], euler_angles[1], euler_angles[2], size[0], size[1], size[2]))


        pts__, _ = project_points(object_bbox_3d, pose_, K)
        bbox_img_ = draw_bbox_3d(img, pts__, (0,0,255))
        bbox_img_ = cv2.resize(bbox_img_, (640, 480))
        cv2.imshow('Camera Frame', np.hstack((bbox_inter,bbox_img_)))
        #imsave(f'{str(output_dir)}/images_out_smooth/{que_id}-bbox.jpg', bbox_img_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

    #cmd=[args.ffmpeg, '-y', '-framerate','30', '-r', '30',
    #     '-i', f'{output_dir}/images_out_smooth/%d-bbox.jpg',
    #     '-c:v', 'libx264','-pix_fmt','yuv420p', f'{output_dir}/video.mp4']
    #subprocess.run(cmd)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain.yaml')
    parser.add_argument('--database', type=str, default="custom/part1")
    #parser.add_argument('--output', type=str, default="data/custom/mouse_processed/test")

    # input video process
    #parser.add_argument('--video', type=str, default="mouse-test.mp4")
    parser.add_argument('--resolution', type=int, default=960)
    #parser.add_argument('--transpose', action='store_true', dest='transpose', default=False)

    # smooth poses
    parser.add_argument('--num', type=int, default=15)
    parser.add_argument('--std', type=float, default=2.5)

    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
    args = parser.parse_args()
    main(args)