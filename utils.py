import yaml
import numpy as np
import cv2 as cv
from scipy import ndimage
import open3d as o3d
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import denseCRF



def read_pose(dir, field = 'robot'):
    """
    The 'robot' field saves the hand frame w.r.t. base frame (base2hand), 
    and following claculation of transformation hand2cam is needed.
    """
    with open(dir, 'r') as file:
        samples = yaml.safe_load(file)['samples']
    poses = np.ones((len(samples),7), dtype=np.float32)
    for i, sample in zip(range(len(samples)), samples):
        translation = [sample[field]['translation']['x'],
                        sample[field]['translation']['y'],
                        sample[field]['translation']['z']]
        quat = [sample[field]['rotation']['x'],
                sample[field]['rotation']['y'],
                sample[field]['rotation']['z'],
                sample[field]['rotation']['w']]
        poses[i] = np.asarray(translation + quat, dtype=np.float32)
    return poses

def read_intrinsic(dir):
    CameraIntrinsic = {}
    with open(dir, 'r') as file:
        param = yaml.safe_load(file)
        CameraIntrinsic['image_height'] = param['height']
        CameraIntrinsic['image_width'] = param['width']
        k = param['k']
    CameraIntrinsic['fx'] = k[0]
    CameraIntrinsic['fy'] = k[4]
    CameraIntrinsic['cx'] = k[2]
    CameraIntrinsic['cy'] = k[5]
    return CameraIntrinsic

def DBSCAN_segmentation(pcd, eps=2, min_points=100):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    max_label = labels.max()
    lab_num = []
    for i in range(labels.max()+1):
        lab_num.append(np.sum(labels==i))
    max_label = np.argsort(lab_num)[-1]

    pcd_dbscan = pcd.select_by_index(np.where(labels==max_label)[0])

    # o3d.visualization.draw_geometries([pcd])

    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

    return pcd_dbscan

def get_largest_one_component(img, print_info = False, threshold = None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 2D image
        threshold: a size threshold
    outputs:
        out_img: the output image 
    """
    s = ndimage.generate_binary_structure(2,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img, labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*2 > max_size1):
                component1 = (component1 + component2) > 0
            out_img = component1
    return np.asarray(out_img)

def fill_holes(img):
    """
    filling small holes of a binary volume with morphological operations
    """
    neg = 1 - img
    s = ndimage.generate_binary_structure(2,1) # iterate structure
    labeled_array, numpatches = ndimage.label(neg,s) # labeling
    sizes = ndimage.sum(neg,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component

def project2plane(pcd, plane_model):
    """
    projecting the point cloud data to a 2D plane defined by plane_model (ax+by+cz+d=0)

    Param
    ------
    Input:
        pcd: open3d.geometry.Pointcloud(). Point cloud data to be projected.
        plane_model: list of coefficients with size of 4. (ax+by+cz+d=0)
    """

    # randomly selecting 3 points on the plane. TODO: if the plane orthogonal to the X-O-Y plane
    assert np.abs(plane_model[2]) > 1e-3, "Plane model orthogonal to the X-O-Y plane"
    plane_model = np.array(plane_model, dtype=np.float32)
    random_xy_values = np.random.rand(3,2) * 10 
    random_xy_values_homo = np.concatenate([random_xy_values,np.ones((3,1))], axis=1)
    z_values = -1 * (random_xy_values_homo @ plane_model[[0,1,3], np.newaxis])/plane_model[2]
    basis_points = np.concatenate([random_xy_values, z_values], axis=1)
    
    # projecting the pcd points
    points = np.asarray(pcd.points, dtype=np.float32)
    basis_vectors = [basis_points[0] - basis_points[1], basis_points[0] - basis_points[2]]
    basis_vectors = np.asarray(basis_vectors).T # shape of (3 , 2)
    points_shift = points.T - basis_points[0][:,np.newaxis]
    coeff = np.linalg.lstsq(basis_vectors, points_shift) # shape of (2, N)
    projected_points = basis_vectors@coeff # shape of (3, N)
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)# - np.mean(xyz_ROI,axis=0)) 

    return projected_pcd

def compute_normal(plane_pdc):
    """
    computing the norm vectos of the plane region.

    Param
    ------
    Input:
        plane_pdc: open3d.geometry.Pointcloud
    """
    plane_pdc.estimate_normals()
    normals = np.asarray(plane_pdc.normals)

    normals[normals[:,2]>0] *= -1 
    normal = np.mean(normals, axis=0)
    normal = normal/np.linalg.norm(normal)

    return normal

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())
    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def dense_crf(**kwargs):
        """
        Using dense CRF to refine the segmentation map for anti-aliasing.

        Param
        -----
        Input:
            img     :  a numpy array of shape [H, W, C], where C should be 3.
                        type of img should be np.uint8, and the values are in [0, 255]
            mask    : a probability map of shape [H, W, L], where L is the number of classes
                        type of P should be np.float32
        
        Param:
            param   : a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                w1    :   weight of bilateral term, e.g. 10.0
                alpha :   spatial distance std, e.g., 80
                beta  :   rgb value std, e.g., 15
                w2    :   weight of spatial term, e.g., 3.0
                gamma :   spatial distance std for spatial term, e.g., 3
                it    :   iteration number, e.g., 5
        """
        img = kwargs['image']
        mask = kwargs['mask']
        Lq = np.asarray(mask, dtype=np.float32)
        Background = Lq>=2
        
        Lq = np.concatenate([Background[:,:,np.newaxis], (Background == False)[:,:,np.newaxis]], axis= 2)
        Lq = np.asarray(Lq, dtype=np.float32)
        plt.imshow(Lq[:,:,1])
        plt.show()
        plt.imshow(Lq[:,:,0])
        plt.show()
        
        prob = Lq[:,:,:2]
        # prob[:,:,0] = 1.0 - prob[:,:,0]
        w1    = 10.0  # weight of bilateral term
        alpha = 80    # spatial std
        beta  = 13    # rgb  std
        w2    = 3.0   # weight of spatial term
        gamma = 3     # spatial std
        it    = 5.0   # iteration
        param = (w1, alpha, beta, w2, gamma, it)
        lab = denseCRF.densecrf(img, prob, param)
        
        lab = Image.fromarray(lab)
        lab = np.asarray(lab, dtype = np.float32)
        lab = np.asarray(lab!=1, dtype = np.float32)

        
        return lab

def smooth_raster_lines(self, im, filterRadius, filterSize, sigma):
    smoothed = np.zeros_like(im)
    contours, hierarchy = cv.findContours(im, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    for countur_idx, contour in enumerate(contours):
        len_ = len(contour) + 2 * filterRadius
        idx = len(contour) - filterRadius

        x = []
        y = []    
        for i in range(len_):
            x.append(contour[(idx + i) % len(contour)][0][0])
            y.append(contour[(idx + i) % len(contour)][0][1])

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        xFilt = cv.GaussianBlur(x, (filterSize, filterSize), sigma, sigma)
        xFilt = [q[0] for q in xFilt]
        yFilt = cv.GaussianBlur(y, (filterSize, filterSize), sigma, sigma)
        yFilt = [q[0] for q in yFilt]


        smoothContours = []
        smooth = []
        for i in range(filterRadius, len(contour) + filterRadius):
            smooth.append([xFilt[i], yFilt[i]])

        smoothContours = np.asarray([smooth], dtype=np.int32)

        color = (0,0,0) if hierarchy[countur_idx][3] > 0 else (255,255,255)
        cv.drawContours(smoothed, smoothContours, 0, color, -1)
    plt.imshow(smoothed)
    plt.show()
    return(smoothed)

def erosion(**param):
    mask = param['mask']
    mask = Image.fromarray(mask==2)
    # mask = mask.filter(ImageFilter.ModeFilter(9))
   
    mask = mask.filter(ImageFilter.MaxFilter(3))
    mask = mask.filter(ImageFilter.ModeFilter(9))
    mask = np.asarray(mask, dtype=np.float32)
        
    # mask[mask != 1]=1
    # mask_foreground = get_largest_one_component(mask!=1,1)
    # mask[mask_foreground != 1]=1
    
    return mask

post_processes = {'erosion': erosion, 
                    'denseCRF': dense_crf}
if __name__ ==  '__main__':
    read_pose('/home/sean/laser_ws/data/poses.yaml')
    print(read_intrinsic('/home/sean/laser_ws/data/intrinsic.yaml'))