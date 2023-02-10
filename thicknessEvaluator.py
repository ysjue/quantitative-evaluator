import numpy as np
import zivid
from sample_utils.display import display_depthmap, display_pointcloud, display_rgb
from PIL import Image, ImageFilter
from utils import compute_normal, project2plane, DBSCAN_segmentation, post_processes
import open3d as o3d
from matplotlib import pyplot as plt
import denseCRF
import cv2 as cv



class thicknessEvaluator(object):
    def __init__(self, file_dir:str = '', automatic_mode:bool = False, lateral_step_num: int=100):
        self.file_dir = file_dir
        self.automatic_mode = automatic_mode
        self.lateral_step_num = lateral_step_num
        self.homo = lambda X: np.concatenate([X,np.ones((1,X.shape[1]))], axis=0) # X is of shape (3, N)

    def postprocess_gel_mask(self,img, mask, method = 'erosion'):

        post = post_processes[method]
        param = {'image':img, 'mask':mask}
        mask_post = post(**param)
        fig = plt.figure()
        plt.subplot(1,3,1); plt.axis('off'); plt.imshow(img); plt.title('input image')
        plt.subplot(1,3,2); plt.axis('off'); plt.imshow(mask); plt.title('initial label')
        plt.subplot(1,3,3); plt.axis('off'); plt.imshow(mask_post); plt.title('after '+ method)
        plt.show()

        return mask_post

    
    def read_gel_mask(self, mask_path):
        mask = Image.open(mask_path)
  
        mask = np.asarray(mask, dtype=np.float32)

        mask[mask>=2] = 2
        mask = self.postprocess_gel_mask(self.image,mask,method='erosion')
        mask = np.array(mask, dtype= np.float32)
        mask = np.float32(mask==1)*2
        self.mask = mask
        
        return mask

    def read_color_image(self, image_path):
        image = Image.open(image_path)
        self.image = np.asarray(image, dtype = np.uint8)
        return self.image 
    
    def read_plane_model(self, plane_path):
        """
        Read the plane point cloud data (.ply) from the plane_path.
        """
        plane_pcd = o3d.io.read_point_cloud(plane_path)
        plane_model, _ = plane_pcd.segment_plane(distance_threshold=0.07, ransac_n=4, num_iterations=1000)
        self.plane_model = np.asarray(plane_model, dtype=np.float32)
        return plane_model

    def project2plane(self, pcd, plane_model):
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
        coeff = np.linalg.lstsq(basis_vectors, points_shift, rcond=None)[0] # shape of (2, N)
        projected_points = basis_vectors@coeff +  basis_points[0][:,np.newaxis] # shape of (3, N)
        projected_pcd = o3d.geometry.PointCloud()
        idxes = [idx for idx in range(projected_points.shape[1]) \
                                        if not np.any(np.isnan(projected_points[:,idx]))]
        projected_points = projected_points[:,idxes]
        projected_points = np.asarray(projected_points)

        projected_pcd.points = o3d.utility.Vector3dVector(projected_points.T)

        return projected_pcd


    def thickness_eval(self, projected_pcd):
        
        projected_points = np.asarray(projected_pcd.points).T # shape of (3, N)
        projected_points_shifted = projected_points - np.mean(projected_points, axis = 1, keepdims = True)
        u,s,vh = np.linalg.svd(projected_points_shifted.T)
        direc_vec = vh[0]/np.linalg.norm(vh[0])
        norm_vec = self.plane_model[:3]/np.linalg.norm(self.plane_model[:3])
        
        Rot = np.array([vec/np.linalg.norm(vec) for vec in [np.cross(direc_vec, norm_vec), direc_vec, norm_vec]]).T
        
        transition = np.mean(projected_points, axis = 1)
    
        transform_origin_plane = np.concatenate([ np.concatenate([Rot, transition[:,np.newaxis]],axis = 1),\
                                            np.array([[0,0,0,1]])], axis = 0)
        transform_plane_origin = np.linalg.inv(transform_origin_plane)
        proj_points_plane = transform_plane_origin @ self.homo(projected_points)
        
        idxes = np.argsort(proj_points_plane[1])
        proj_points_plane = np.asarray([proj_points_plane[:3,idx] for idx in idxes], dtype=np.float32).T # shape of (3, N)
        lateral_values = proj_points_plane[1]
        thickness_dirc_values = proj_points_plane[0]
        gel_length = np.max(lateral_values) - np.min(lateral_values)
        step_size = gel_length * 1.0/self.lateral_step_num
        thickness = []
        median_points = []
        for i in range(self.lateral_step_num):
            lateral_seg_start = proj_points_plane[1,0] + i * step_size
            lateral_seg_end = lateral_seg_start + step_size
            idxes = np.where((lateral_values>=lateral_seg_start) *  (lateral_values <= lateral_seg_end))[0]
            median_points.append([0,0.5*(lateral_seg_start+lateral_seg_end),0]) # w.r.t. the plane frame
            if len(idxes) < 3:
                thickness.append(0)
                continue
            thickness.append(np.max(thickness_dirc_values[idxes]) -  np.min(thickness_dirc_values[idxes]))
        thickness = np.asarray(thickness)
        min_seg_idx,max_seg_idx= np.argsort(thickness)[[0,-1]]
        
        # visulization
        proj_plane = o3d.geometry.PointCloud()
        proj_plane.points = o3d.utility.Vector3dVector(proj_points_plane.T)
        median_pdc = o3d.geometry.PointCloud()
        median_pdc.points = o3d.utility.Vector3dVector(np.asarray(median_points))
        extreme_pdc = o3d.geometry.PointCloud()
        extreme_pdc.points = o3d.utility.Vector3dVector(np.asarray(np.asarray(median_points)[[min_seg_idx,max_seg_idx]]))
        proj_plane.paint_uniform_color([1, 0.706, 0])
        median_pdc.paint_uniform_color([1, 0, 0])
        extreme_pdc.paint_uniform_color([0, 0, 1])
        
        # for thickness 
        o3d.visualization.draw_geometries([ proj_plane,median_pdc,extreme_pdc])
        self.thickness = thickness
        return thickness

    def read_pcd_from_zdf(self, data_file):
        with zivid.Application():
            frame = zivid.Frame(data_file)
            point_cloud = frame.point_cloud()
            xyz = point_cloud.copy_data("xyz")
            rgba = point_cloud.copy_data("rgba")
            # display_rgb(rgba[:, :, 0:3], title="RGB image")
            # display_depthmap(xyz)
            # display_pointcloud(xyz, rgba[:, :, 0:3])
            self.xyz = xyz
            self.rgba = rgba
            return xyz, rgba
        
    
        


    def mask_point_cloud(self, xyz:np.ndarray, mask:np.ndarray, label:int = 2):
        """
        Read point cloud data and apply a binary mask.

        Param
        ------
        Inputs: 
            xyz: ndarray with shpae H x W x 3. xyz information acquired from .zdf file
            mask: ndarray with shape (H x W). Binary mask.
            label: int. Background label
        Outputs:
            pcd: o3d.geometry.Pointcloud
        """

        print("Masking point cloud")
        xyz_masked = xyz.copy()
        xyz_masked[mask == label] = np.nan


        # display_depthmap(xyz_masked)
        mask_rgba = self.rgba.copy()
        # mask_rgba[mask==label] = np.array([255,255,255,255])
        mask_rgba[mask!=label] = np.array([0,0,255,255])
        display_pointcloud(xyz,mask_rgba[:, :, 0:3])

        display_pointcloud(xyz_masked, self.rgba[:, :, 0:3])

        xyz_ROI = xyz_masked[mask!=label]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_ROI)# - np.mean(xyz_ROI,axis=0))
        pcd.paint_uniform_color([1, 0, 0])
        print(len(np.asarray(pcd.points)))
        # pcd = DBSCAN_segmentation(pcd,eps=0.45, min_points=10)
        
        o3d.visualization.draw_geometries([pcd]) 
        o3d.io.write_point_cloud('visualization.ply',pcd)
        return pcd

    def pcd_segmentation(self):
        try: 
            xyz = self.xyz
        except:
            print("Failed to read xyz data")
        try: 
            mask = self.mask
            
            # plt.imshow(mask)
            # plt.show()
        except:
            print("Failed to read mask data")

        return self.mask_point_cloud(xyz, mask, label=2)
    



if __name__ == "__main__":
    evaluator = thicknessEvaluator()
    # read color image
    evaluator.read_color_image('D:/Bioprinting/data/data_1_26_2023/frame1.jpg')

    evaluator.read_gel_mask('D:/Bioprinting/data/data_1_26_2023/annotation_best1.jpg')
    evaluator.read_pcd_from_zdf('D:/Bioprinting/data/data_1_26_2023/1.zdf')

    
    
    # evaluator.read_gel_mask('D:/Bioprinting/data/camera_accuracy/annotation_2mm5.jpg')
    # evaluator.read_pcd_from_zdf('D:/Bioprinting/data/camera_accuracy/zivid5.zdf')
    pcd = evaluator.pcd_segmentation()
    o3d.visualization.draw_geometries([pcd]) 

    evaluator.read_plane_model('D:/Bioprinting/data/data_1_26_2023/plane1.ply')
    # evaluator.read_plane_model('D:/Bioprinting/data/camera_accuracy/plane5.ply')
    projected_pcd = evaluator.project2plane(pcd, evaluator.plane_model)
    pt = np.array(projected_pcd)
    o3d.visualization.draw_geometries([ projected_pcd])
    thickness = evaluator.thickness_eval(projected_pcd)
    
    thickness = thickness[10:-10]
    plt.plot([i for i in range(len(thickness))], thickness)
    
    print(np.mean(thickness),np.max(thickness),np.min(thickness),np.std(thickness))
    print([t  for t in thickness ])
    plt.show()
