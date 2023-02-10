import open3d as o3d
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
"""
Segmentation of the biogel strains based on the 3D point cloud data

"""
def z_range_filter(pcd, z_range = [565,577.9] ):
    points_array = np.asarray(pcd.points, dtype=np.float32)
    z_vals = points_array[:,2]
    idxes = list(np.where(z_vals < z_range[0])[0]) + list(np.where(z_vals > z_range[1])[0])
    return pcd.select_by_index(idxes,invert=True)

def DBSCAN_segmentation(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=1.2, min_points=15))
    
    max_label = labels.max()
    lab_num = []
    for i in range(labels.max()):
        lab_num.append(np.sum(labels==i))
    max_label = np.argsort(lab_num)[-1]

    pcd_dbscan = pcd.select_by_index(np.where(labels==max_label)[0])

    # o3d.visualization.draw_geometries([pcd])

    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])


    X = np.concatenate([np.array(pcd.colors),np.array(pcd.points)[:,2:3]],axis=1)
    X = np.array(pcd.colors)
    kmeans = KMeans(n_clusters=2).fit(X)
    o3d.visualization.draw_geometries([pcd_dbscan])

    pcd1 = pcd.select_by_index(np.where(kmeans.labels_ == 1)[0])
    pcd2 = pcd.select_by_index(np.where(kmeans.labels_ == 0)[0])
    o3d.visualization.draw_geometries([pcd1])

    return pcd_dbscan

# def color_clustering(pcd)

def compute_normal(plane_pdc):
    """
    compute the norm vectos of the plane region.
    """
    plane_pdc.estimate_normals()
    normals = np.asarray(plane_pdc.normals)

    normals[normals[:,2]>0] *= -1 
    normal = np.mean(normals, axis=0)
    normal = normal/np.linalg.norm(normal)

    return normal

def project(pcd, normal, plane_model):
    plane_model = np.array(plane_model)[None,:]
    points = np.asarray(pcd.points).T # 3 x n
    denom = np.matmul(plane_model[:,:-1], normal[:,None])[0]
    numer = -plane_model[:,-1] - np.matmul(plane_model[:,:-1], points)[0]
    coeff = numer/denom
    assert len(coeff) == points.shape[1]
    projected_points = points.T + np.array([c * normal for c in coeff])
    pcd_projected = pcd
    pcd_projected.points = o3d.utility.Vector3dVector(projected_points)
    return pcd_projected
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3d segmentation')
    parser.add_argument('--image_dir', type=str, default="data\pcd_zivid\sample2_3mm.ply")
    config = parser.parse_args()

    pcd = o3d.io.read_point_cloud(config.image_dir)
    # pcd = z_range_filter(pcd)

    o3d.visualization.draw_geometries([pcd])


    plane_model, inliers = pcd.segment_plane(distance_threshold=0.07, ransac_n=4, num_iterations=1000)
    
    plane_cloud = pcd.select_by_index(inliers)
    o3d.visualization.draw_geometries([plane_cloud])
    normal = compute_normal(plane_cloud)


    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    # inlier_cloud = outlier_cloud.select_by_index(inliers)
    # outlier_cloud1 = outlier_cloud.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([outlier_cloud])

    seg_pcd = DBSCAN_segmentation(outlier_cloud)
    o3d.io.write_point_cloud('seg_pcd_sample2_3mm.ply', seg_pcd)
    projected_pcd = project(seg_pcd, normal, plane_model)
    o3d.io.write_point_cloud('projected_seg_pcd_sample2_3mm.ply', projected_pcd)
    o3d.visualization.draw_geometries([seg_pcd, projected_pcd])





