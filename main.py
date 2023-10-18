###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    U, S, V_trans = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]])
    Q1 = U.dot(W).dot(V_trans)
    Q2 = U.dot(W.T).dot(V_trans)
    R1 = np.linalg.det(Q1)*Q1
    R2 = np.linalg.det(Q2)*Q2
    T1 = np.array(U[:,2])
    T1 = T1[:,np.newaxis]
    T2 = -T1 
    # return np.hstack((R1, T1))
    return [np.hstack((R1, T1)),np.hstack((R1, T2)),np.hstack((R2, T1)),np.hstack((R2, T2))]
    # raise Exception('Not Implemented Error')

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    equ_matrix = []
    for i in range(len(camera_matrices)):
        u,v = image_points[i]
        projection_matrix = camera_matrices[i]
        equ_matrix.append(np.hstack([v * projection_matrix[2, :] - projection_matrix[1, :]]))
        equ_matrix.append(np.hstack([projection_matrix[0, :] - u * projection_matrix[2, :]]))
    
    equ_matrix = np.array(equ_matrix)
    U, S, V_trans = np.linalg.svd(equ_matrix)
    point_3d = V_trans[-1, :]
    point_3d /= point_3d[-1]
    # print(point_3d)
    return point_3d[:-1]
    # raise Exception('Not Implemented Error')

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    error = []
    P = np.hstack((point_3d, [1]))
    for i in range(len(camera_matrices)):
        u, v = image_points[i]
        projection_matrix = camera_matrices[i]
        projected_point = projection_matrix.dot(P)
        x, y, z = projected_point
        error.append(x/z - u)
        error.append(y/z - v)
    # print("error\n", error)
    return error
    # raise Exception('Not Implemented Error')

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    jacobian = []
    P = np.hstack((point_3d, [1]))
    delta = 0.0001
    for i in range(len(camera_matrices)):
        projection_matrix = camera_matrices[i]
        projected_point = projection_matrix.dot(P)
        projected_point /= projected_point[2]
        error = []
        for j in range(2):
            error.append(projected_point[j] - point_3d[j])
        
        row1 = []
        row2 = []
        near_px = np.hstack((point_3d, [1]))
        near_px[0] += delta
        near_pointx = projection_matrix.dot(near_px)
        near_pointx /= near_pointx[2]
        row1.append((near_pointx[0]-error[0]-point_3d[0])/delta)
        row2.append((near_pointx[1]-error[1]-point_3d[1])/delta)

        near_py = np.hstack((point_3d, [1]))
        near_py[1] += delta
        near_pointy = projection_matrix.dot(near_py)
        near_pointy /= near_pointy[2]
        row1.append((near_pointy[0]-error[0]-point_3d[0])/delta)
        row2.append((near_pointy[1]-error[1]-point_3d[1])/delta)

        near_pz = np.hstack((point_3d, [1]))
        near_pz[2] += delta
        near_pointz = projection_matrix.dot(near_pz)
        near_pointz /= near_pointz[2]
        row1.append((near_pointz[0]-error[0]-point_3d[0])/delta)
        row2.append((near_pointz[1]-error[1]-point_3d[1])/delta)
        jacobian.append(row1)
        jacobian.append(row2)

    jacobian = np.array(jacobian)
    # print("jacobian\n",jacobian)
    return jacobian
    # raise Exception('Not Implemented Error')
        
'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    point_3d = linear_estimate_3d_point(image_points, camera_matrices)
    for i in range(10):
        jacobian_matrix = jacobian(point_3d, camera_matrices)
        error = reprojection_error(point_3d, image_points, camera_matrices)
        point_3d = point_3d - (np.linalg.inv(jacobian_matrix.T.dot(jacobian_matrix))).dot(jacobian_matrix.T).dot(error)
    # print("nonlinear_point_3d\n",nonlinear_point_3d)
    return point_3d
    # raise Exception('Not Implemented Error')

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    RT_array = estimate_initial_RT(E)
    max_count = 0
    RT_index = 0
    for index, RT in enumerate(RT_array):
        camera_matrices = np.zeros((2, 3, 4))
        camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
        camera_matrices[1, :, :] = K.dot(RT)
        count = 0
        for points in image_points:
            for i in range(len(points)):
                for j in range(i+1,len(points)):
                    estimate_3d_point = nonlinear_estimate_3d_point([points[i], points[j]], camera_matrices)
                    if estimate_3d_point[2] > 0:
                        count += 1
                    # estimate_3d_point = nonlinear_estimate_3d_point([points[j], points[i]], camera_matrices)
                    # if estimate_3d_point[2] > 0:
                    #     count += 1
        
        if count > max_count:
            max_count = count
            RT_index = index 
    
    return RT_array[RT_index]
    # raise Exception('Not Implemented Error')

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    # print("expected_jacobian\n", expected_jacobian)
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')
# 
# python3 PtsVisualizer/visualize.py results.npy 