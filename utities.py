import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from skimage import io
import os
import json
import pdb
import copy
from scipy import ndimage
import cv2

import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()



def read_rgb(rgb_file):
    rgb = io.imread(rgb_file)
    # plt.imshow(rgb)
    # plt.title(rgb_file)
    # plt.show()
    return rgb

def read_depth(depth_file):
    depth = io.imread(depth_file)
    # Reference: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
    depth = depth[:, :, 0] * 1.0 + depth[:, :, 1] * 256.0 + depth[:, :, 2] * (256.0 * 256)
    depth = depth * (1/ (256 * 256 * 256 - 1))
    # plt.imshow(depth)
    # plt.title(depth_file)
    # plt.show()
    return depth

def point_cloud_to_image(points,color ,K,transformation = None):
    points = np.transpose(points, (1,0))
    if transformation is not None:
        tmp = np.ones((4,points.shape[1]))
        tmp[:3,:] = points
        tmp = transformation @ tmp
    else:
        tmp = points
    tmp = K @ tmp
    tmp1 = tmp/tmp[2,:]
    # Note that multiple points might be mapped to the same pixel
    # The one with the lowest depth value should be assigned to that pixel
    # However, note this has not been implemented here
    # One may want to implement this
    u_cord = np.clip(np.round(tmp1[0,:]),0,511).astype(int)
    v_cord = np.clip(np.round(tmp1[1,:]),0,511).astype(int)
    if color is not None:
        imtmp = np.zeros((512,512,3)).astype(np.uint8)
        imtmp[u_cord, v_cord,:]= (color * 255).astype(np.uint8)
        
    else:
        imtmp = np.zeros((512,512)).astype(np.uint8)
        imtmp[u_cord, v_cord] = tmp[2,:]
        
    imtmp = cv2.flip(ndimage.rotate(imtmp, 90),1) # For some reason the axis were flipped
                                                  # therefore have been fixed here
        
    # plt.imshow(imtmp)
    # plt.show()
        
    return imtmp
    

def depth_to_local_point_cloud(depth, color=None, k = np.eye(3),max_depth=1.0):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the 3D position (relative to the camera) of each pixel and its corresponding
    RGB color of an array.
    "max_depth" is used to omit the points that are far enough.
    """
    "Reference: https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/carla/image_converter.py"
    far = 1000.0  # max depth in meters.
    normalized_depth = depth# depth_to_array(image)
    height, width = depth.shape

    # 2d pixel coordinates
    pixel_length = width * height
    u_coord = repmat(np.r_[width-1:-1:-1],
                     height, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[height-1:-1:-1],
                     1, width).reshape(pixel_length)
    if color is not None:
        color = color.reshape(pixel_length, 3)
    normalized_depth = np.reshape(normalized_depth, pixel_length)

    # Search for pixels where the depth is greater than max_depth to
    # delete them
    max_depth_indexes = np.where(normalized_depth > max_depth)
    normalized_depth = np.delete(normalized_depth, max_depth_indexes)
    u_coord = np.delete(u_coord, max_depth_indexes)
    v_coord = np.delete(v_coord, max_depth_indexes)
    if color is not None:
        color = np.delete(color, max_depth_indexes, axis=0)

    # pd2 = [u,v,1]
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

    # P = [X,Y,Z]
    p3d = np.dot(np.linalg.inv(k), p2d)
    p3d *= normalized_depth * far
    
    p3d = np.transpose(p3d, (1,0))

    if color is not None:
        return p3d, color / 255.0
    else:
        return p3d, None


# FOLDER = "/usr/prakt/s0013/week1/dataset/"

# #generate RGB5 6 3 4 0 1 2
# count = 0
# for i in range(0,400,4):
#     cam1 = 2
#     frame1 = i
#     cam2 = 1
#     frame2 = i

#     frame1 = format(frame1, '05d')
#     frame2 = format(frame2, '05d')

#     rgb_file1   = os.path.join(FOLDER, "CameraRGB{}/image_{}.png".format(cam1, frame1))
#     rgb_file2   = os.path.join(FOLDER, "CameraRGB{}/image_{}.png".format(cam2, frame2))
#     depth_file1 = os.path.join(FOLDER, "CameraDepth{}/image_{}.png".format(cam1, frame1))
#     depth_file2 = os.path.join(FOLDER, "CameraDepth{}/image_{}.png".format(cam2, frame2))
#     intrinsics_file = os.path.join(FOLDER,"camera_intrinsic.json")

#     rgb1 = read_rgb(rgb_file1)
#     rgb2 = read_rgb(rgb_file2)
#     depth1 = read_depth(depth_file1)
#     depth2 = read_depth(depth_file2)

#     with open(intrinsics_file) as f:
#         K = json.load(f)
#     K = np.array(K)
#     pc1, color1 = depth_to_local_point_cloud(depth1, color=rgb1, k = K,max_depth=0.05)
#     #pc2, color2 = depth_to_local_point_cloud(depth2, color=rgb2, k = K,max_depth=0.05)
#     # print(pc1)
#     # rgb1_projected = point_cloud_to_image(pc1,color1, K).astype(np.int16)
#     # plt.imshow(np.abs(rgb1_projected - rgb1))

#     #projected_image = point_cloud_to_image(pc1,color1, K, transformation = transformation).astype(np.int16)
#     #plt.imshow(np.abs(rgb2-projected_image))
#     transformation1 = np.eye(4)[:3,:]
#     print(transformation1)
#     transformation1[0,3] = 0

    

#     projected_image = point_cloud_to_image(pc1,color1, K, transformation = transformation1).astype(np.int16)
#     #p2 = point_cloud_to_image(pc2,color2, K, transformation = transformation2).astype(np.int16)
#     #projected_image = (np.abs(p1-p2))/1.0
#     cv2.imwrite('/usr/prakt/s0013/week1/dataset/CameraRGB20/image_'+str(count)+'.png', projected_image)
#     count+=1


