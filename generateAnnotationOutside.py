from math import radians
from operator import gt
import os
import re
import json
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from unitConversion import *

intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']

def annotate_outside():
    print("Starting outside annotation...")
    # read boxes for each frame, and save
    _, gt_3d = read_gt_outside_2D3D_percam(0)
    num_of_frame = int(max(gt_3d[:, 0])) + 1

    # all_boxes[frame_num][cam] => (N_of_outsider, 4)
    all_boxes_and_xy = [[[] for j in range(6)] for i in range(num_of_frame)] 

    for cam in range(6):
        # obtain camera matrix
        fp_calibration = cv2.FileStorage(f'calibrations/intrinsic/{intrinsic_camera_matrix_filenames[cam]}',
                                            flags=cv2.FILE_STORAGE_READ)
        cameraMatrix, distCoeffs = fp_calibration.getNode('camera_matrix').mat(), fp_calibration.getNode(
                'distortion_coefficients').mat()
        fp_calibration.release()
        fp_calibration = cv2.FileStorage(f'calibrations/extrinsic/{extrinsic_camera_matrix_filenames[cam]}',
                                            flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

        gt_2d, gt_3d = read_gt_outside_2D3D_percam(cam)
        assert gt_2d.shape[0] == gt_3d.shape[0]
        
        coord_x, coord_y = gt_3d[:, 1], gt_3d[:, 2]
        centers3d = np.stack([coord_x, coord_y, np.zeros_like(coord_y)], axis=1)
        points3d8s = []
        points3d8s.append(centers3d + np.array([MAN_RADIUS, MAN_RADIUS, 0]))
        points3d8s.append(centers3d + np.array([-MAN_RADIUS, MAN_RADIUS, 0]))
        points3d8s.append(centers3d + np.array([MAN_RADIUS, -MAN_RADIUS, 0]))
        points3d8s.append(centers3d + np.array([-MAN_RADIUS, -MAN_RADIUS, 0]))
        points3d8s.append(centers3d + np.array([MAN_RADIUS, MAN_RADIUS, MAN_HEIGHT]))
        points3d8s.append(centers3d + np.array([-MAN_RADIUS, MAN_RADIUS, MAN_HEIGHT]))
        points3d8s.append(centers3d + np.array([MAN_RADIUS, -MAN_RADIUS, MAN_HEIGHT]))
        points3d8s.append(centers3d + np.array([-MAN_RADIUS, -MAN_RADIUS, MAN_HEIGHT]))
        bbox = np.ones([centers3d.shape[0], 4]) * np.array([IMAGE_WIDTH, IMAGE_HEIGHT, 0, 0])  # xmin,ymin,xmax,ymax
        for i in range(8):  # for all 8 points
            points_img, _ = cv2.projectPoints(points3d8s[i], rvec, tvec, cameraMatrix, distCoeffs)
            points_img = points_img.squeeze()
            bbox[:, 0] = np.min([bbox[:, 0], points_img[:, 0]], axis=0)  # xmin
            bbox[:, 1] = np.min([bbox[:, 1], points_img[:, 1]], axis=0)  # ymin
            bbox[:, 2] = np.max([bbox[:, 2], points_img[:, 0]], axis=0)  # xmax
            bbox[:, 3] = np.max([bbox[:, 3], points_img[:, 1]], axis=0)  # xmax
            pass
        points_img, _ = cv2.projectPoints(centers3d, rvec, tvec, cameraMatrix, distCoeffs)
        points_img = points_img.squeeze()
        bbox[:, 3] = points_img[:, 1]
        bbox_with_idx = np.concatenate((gt_3d[:, :1], bbox), axis=1)
        bbox_with_idx = np.array(bbox_with_idx, dtype=int)
        for frame in range(num_of_frame):
            # limit all the data within the curent frame
            mask = [gt_2d[:, 0] == frame][0]

            bbox_with_idx = bbox_with_idx[mask, :]
            gt_2d = gt_2d[mask, :]
            gt_3d = gt_3d[mask, :]
            points_img = points_img[mask, :]

            all_boxes_and_xy[frame][cam] = np.concatenate((bbox_with_idx[:, 1:], gt_2d[:, 1:3], gt_3d[:, -3:]), axis=1)
            # # visualize
            # img = cv2.imread(f"D:\\2.study\\MultiviewX\\Image_subsets\\C{cam + 1}\\0000.png")
            # for idx, box in enumerate(bbox_with_idx):
            #     xmin, ymin, xmax, ymax = box[1:]
            #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color = (0, 0, 255), thickness=2)
            #     cv2.circle(img, (int(gt_2d[idx, 1]),int(gt_2d[idx, 2])), radius=4, thickness=-1, color=(0, 0, 255))
            #     cv2.circle(img, (int(points_img[idx, 0]),int(points_img[idx, 1])), radius=2, thickness=-1, color=(0, 255, 0))
            # cv2.imwrite("dzc.jpg", img)
            # print()

    for i in range(num_of_frame):
        # cam_box (N, 4)
        outside_annot = []
        
        for idx, cam_boxes in enumerate(all_boxes_and_xy[i]):
            # per frame
            cur_cam = {}
            cur_cam["cam_id"] = idx + 1
            outsiders = []

            for id, data in enumerate(cam_boxes):
                xmin, ymin, xmax, ymax, x, y, x_3d, y_3d, z_3d = data
                outsider = {}
                outsider = json.loads(json.dumps(outsider))
                outsider["outside_id"] = id
                outsider["2d"] = [int(x), int(min(y, ymax))]
                outsider["3d"] = [x_3d, y_3d, z_3d]
                outsider["xmin"] = int(xmin)
                outsider["ymin"] = int(ymin)
                outsider["xmax"] = int(xmax)
                outsider["ymax"] = int(ymax)
                outsiders.append(outsider)
            cur_cam["outsiders"] = outsiders
            outside_annot.append(cur_cam)

        with open('annotations_positions/{:05d}_outside.json'.format(i), 'w') as fp:
            json.dump(outside_annot, fp, indent=4)
        if i == 0:
            for cam in range(6):
                img = Image.open(f'Image_subsets/C{cam + 1}/0000.png')
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                outsiders = outside_annot[cam]['outsiders']
                for outsider in outsiders:
                    
                    bbox = tuple([outsider['xmin'], outsider['ymin'], outsider['xmax'], outsider['ymax']])
                    if bbox[0] == -1 and bbox[1] == -1:
                        continue
                    cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0), 2)
                gt_2d_outside, _ = read_gt_outside_2D3D_percam(cam)
                mask = [gt_2d_outside[:, 0] == 0][0]
                gt_2d_outside = gt_2d_outside[mask, :]
                for pt_2d in gt_2d_outside:
                    cv2.circle(img, (int(pt_2d[1]), int(pt_2d[2])), radius=4, color=(0,0,255), thickness=-1)

                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img.save(f'imgs/bbox_cam{cam + 1}_outside.png')
                # plt.imshow(img)
                plt.show()
                pass



def read_gt_outside_2D3D_percam(cam, frame = None):
    """
    return: n * (frame, x, y, [z])
    """
    gt_2d = np.loadtxt(f'matchings/Camera{cam + 1}_outside.txt')
    gt_3d = np.loadtxt(f'matchings/Camera{cam + 1}_3d_outside.txt')

    # return gt with designated frame
    if frame:
        gt_2d = np.concatenate((gt_2d[gt_2d[:, :1] == frame, 0], gt_2d[gt_2d[:, 0] == frame, -2:]), axis = 1)
        gt_3d = np.concatenate((gt_3d[gt_3d[:, :1] == frame, 0], gt_3d[gt_3d[:, 0] == frame, -3:]), axis = 1)
    else:
        gt_2d = np.concatenate((gt_2d[:, :1], gt_2d[:, -2:]), axis = 1)
        gt_3d = np.concatenate((gt_3d[:, :1], gt_3d[:, -3:]), axis = 1)

    # mask the point inside the image, crop a little bit to avoid outliers
    # 

    mask1 = np.where(np.logical_and(gt_2d[:, -2] >= 0, gt_2d[:, -2] <= IMAGE_WIDTH))[0]
    gt_2d = gt_2d[mask1, :]
    gt_3d = gt_3d[mask1, :]

    mask2 = np.where(np.logical_and(gt_2d[:, -1] >= 270, gt_2d[:, -1] <= IMAGE_HEIGHT))[0]
    gt_2d = gt_2d[mask2, :]
    gt_3d = gt_3d[mask2, :]

    # for idx, pt in enumerate(gt_2d):
    #     x, y = pt
    #     if cam == 1:
    #         print(idx, pt)
        
    #     color=(0,0,255)
    #     cv2.circle(img_1, (int(x), int(y)), thickness=-1, radius=5, color=color)
    
    return gt_2d, gt_3d

def read_gt_inside_2D3D(cam):
    vis_num = 0
    # img_1 = cv2.imread(f"D:\\2.study\\MultiviewX\\Image_subsets\\C{cam + 1}\\000{vis_num}.png")
    gt_2d = np.loadtxt(f'Image_subsets/Camera{cam + 1}.txt')
    gt_3d = np.loadtxt(f'matchings/Camera{cam + 1}_3d.txt')

    gt_2d = gt_2d[gt_2d[:, 0] == vis_num, -2:]
    gt_3d = gt_3d[gt_3d[:, 0] == vis_num, -3:]

    # mask the point inside the image, crop a little bit to avoid outliers

    mask1 = np.where(np.logical_and(gt_2d[:, -2] >= 0, gt_2d[:, -2] <= IMAGE_WIDTH))[0]
    gt_2d = gt_2d[mask1, :]
    gt_3d = gt_3d[mask1, :]

    mask2 = np.where(np.logical_and(gt_2d[:, -1] >= 270, gt_2d[:, -1] <= IMAGE_HEIGHT))[0]
    gt_2d = gt_2d[mask2, :]
    gt_3d = gt_3d[mask2, :]

    # for idx, pt in enumerate(gt_2d):
    #     x, y = pt
    #     if cam == 1:
    #         print(idx, pt)
        
    #     color=(0,255,0)
    #     cv2.circle(img_1, (int(x), int(y)), thickness=-1, radius=5, color=color)
    
    return gt_2d, gt_3d

def generate_outside_boxes(cam, frame, rvec, tvec, cameraMatrix, distCoeffs):
    gt_2d, gt_3d = read_gt_outside_2D3D_percam(cam, frame)

    coord_x, coord_y = gt_3d[:, 0], gt_3d[:, 1]

    centers3d = np.stack([coord_x, coord_y, np.zeros_like(coord_y)], axis=1)
    points3d8s = []
    points3d8s.append(centers3d + np.array([MAN_RADIUS, MAN_RADIUS, 0]))
    points3d8s.append(centers3d + np.array([-MAN_RADIUS, MAN_RADIUS, 0]))
    points3d8s.append(centers3d + np.array([MAN_RADIUS, -MAN_RADIUS, 0]))
    points3d8s.append(centers3d + np.array([-MAN_RADIUS, -MAN_RADIUS, 0]))
    points3d8s.append(centers3d + np.array([MAN_RADIUS, MAN_RADIUS, MAN_HEIGHT]))
    points3d8s.append(centers3d + np.array([-MAN_RADIUS, MAN_RADIUS, MAN_HEIGHT]))
    points3d8s.append(centers3d + np.array([MAN_RADIUS, -MAN_RADIUS, MAN_HEIGHT]))
    points3d8s.append(centers3d + np.array([-MAN_RADIUS, -MAN_RADIUS, MAN_HEIGHT]))
    bbox = np.ones([centers3d.shape[0], 4]) * np.array([IMAGE_WIDTH, IMAGE_HEIGHT, 0, 0])  # xmin,ymin,xmax,ymax
    for i in range(8):  # for all 8 points
        points_img, _ = cv2.projectPoints(points3d8s[i], rvec, tvec, cameraMatrix, distCoeffs)
        points_img = points_img.squeeze()
        bbox[:, 0] = np.min([bbox[:, 0], points_img[:, 0]], axis=0)  # xmin
        bbox[:, 1] = np.min([bbox[:, 1], points_img[:, 1]], axis=0)  # ymin
        bbox[:, 2] = np.max([bbox[:, 2], points_img[:, 0]], axis=0)  # xmax
        bbox[:, 3] = np.max([bbox[:, 3], points_img[:, 1]], axis=0)  # xmax
        pass
    points_img, _ = cv2.projectPoints(centers3d, rvec, tvec, cameraMatrix, distCoeffs)
    points_img = points_img.squeeze()
    bbox[:, 3] = points_img[:, 1]

    return bbox


if __name__ == "__main__":
    annotate_outside()