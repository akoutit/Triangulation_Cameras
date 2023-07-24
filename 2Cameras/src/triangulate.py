import cv2
import numpy as np
# Paramètres de calibration de la caméra


camera_matrix1 = np.load("camera_matrix1.npy")

camera_matrix2 = np.load("camera_matrix2.npy")

dist_coeffs1 = np.load("dist_coeffs1.npy")

dist_coeffs2 = np.load("dist_coeffs2.npy")

R = np.load("Rot.npy")

t = np.load("Tr.npy")

square_size = 2.433


def triangulate(u1, v1, u2, v2, K1, dist_coeffs1, K2, dist_coeffs2, R, t):
    # Convert pixel coordinates to normalized image coordinates
    pt1_pixel = np.array([[u1, v1]], dtype=np.float32)
    pt2_pixel = np.array([[u2, v2]], dtype=np.float32)

    # Undistort the pixel coordinates
    pt1_undistorted = cv2.undistortPoints(pt1_pixel, K1, dist_coeffs1, P=K1)
    pt2_undistorted = cv2.undistortPoints(pt2_pixel, K2, dist_coeffs2, P=K2)

    # Homogeneous transformation matrices for camera 1 and camera 2
    T1 = np.eye(4)
    T2 = np.hstack((R, t))

    # Triangulate the 3D point
    A = np.zeros((4, 4))
    A[0] = pt1_undistorted[0, 0, 0] * T1[2] - T1[0]
    A[1] = pt1_undistorted[0, 0, 1] * T1[2] - T1[1]
    A[2] = pt2_undistorted[0, 0, 0] * T2[2] - T2[0]
    A[3] = pt2_undistorted[0, 0, 1] * T2[2] - T2[1]

    _, _, V = np.linalg.svd(A)
    point_3d = V[-1, :3] / V[-1, 3]

    return point_3d

def triangulate_points(u1, v1, u2, v2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, t):
    # Conversion des coordonnées en matrices homogènes
    points1 = cv2.undistortPoints(np.array([[u1, v1]], dtype=np.float32), camera_matrix1, dist_coeffs1, P=camera_matrix1)
    points2 = cv2.undistortPoints(np.array([[u2, v2]], dtype=np.float32), camera_matrix2, dist_coeffs2, P=camera_matrix2)
    

    # Matrice de projection de la première caméra
    proj_mat1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    proj_mat1 = np.dot(camera_matrix1,proj_mat1)
    

    # Matrice de projection de la deuxième caméra
    proj_mat2 = np.hstack((R, t))
    proj_mat2 = np.dot(camera_matrix2,proj_mat2)
    # Triangulation
    points_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, points1, points2)

    # Conversion en coordonnées 3D non normalisées
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    # Extraction des coordonnées 3D
    x = points_3d[:, 0, 0]
    y = points_3d[:, 0, 1]
    z = points_3d[:, 0, 2]

    l = [square_size*x[0],square_size*y[0],square_size*z[0]]
    return l
def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]



def triangulate_(u1, v1, u2, v2, K1, d1, K2, d2, R, t):
    # Convert pixel coordinates to normalized image coordinates

    points1 = cv2.undistortPoints(np.array([[u1, v1]], dtype=np.float32), K1, d1, P=K1)[0][0]
    points2 = cv2.undistortPoints(np.array([[u2, v2]], dtype=np.float32), K2, d2, P=K2)[0][0]



    pt1_norm = np.dot(np.linalg.inv(K1), np.array([points1[0],points1[1],1]))
    pt2_norm = np.dot(np.linalg.inv(K2), np.array([points2[0],points2[1],1]))

    # Homogeneous transformation matrices for camera 1 and camera 2
    T1 = np.eye(4)
    T2 = np.hstack((R, t))

    # Triangulate the 3D point
    A = np.zeros((4, 4))
    A[0] = pt1_norm[0] * T1[2] - T1[0]
    A[1] = pt1_norm[1] * T1[2] - T1[1]
    A[2] = pt2_norm[0] * T2[2] - T2[0]
    A[3] = pt2_norm[1] * T2[2] - T2[1]

    _, _, V = np.linalg.svd(A)
    point_3d = V[-1, :3] / V[-1, 3]

    return point_3d




I = np.eye(3)
O = np.zeros((3,1))


Proj1 = camera_matrix1 @ np.hstack((R,t))
Proj2 = camera_matrix2 @ np.hstack((I,O))

R2 = np.array( [[0, 0, -1],
[0, 1, 0],
[1, 0, 0]])
Tr = np.array([[1],[0],[1]])
K1 = np.array([[1000, 0, 640],
[0, 1000, 480],
[0, 0, 1]])

K2 = np.array([[1000, 0, 320],
[0, 1000, 240],
[0, 0, 1]])

d = np.zeros((1,5))

P1 = K1 @ np.hstack((I,O))
P2 = K1 @ np.hstack((R2,Tr))

l = np.array([0, 0, 5, 1])
# print("u,v :")
# print(P1 @ l)
# print(P1)
# point_3d = triangulate_points(640,480,640,480,K1,d,K1,d,R2,Tr)
# print(point_3d)

# print(DLT(P2, P1, [640,480], [640,480]))
# print(cv2.triangulatePoints(P2,P2, np.array([[[740,480]]]), np.array([[[740,480]]])))
# print(triangulate_(640,480,640,480,K1,d,K1,d,R2,Tr))
# print(DLT(P2, P1, [640,480], [640,480]))
# import matplotlib.pyplot as plt
# # x = np.linspace(0,2*480,1000)
# # Y = []
# # for x_ in x:

# #     y = triangulate_(640, 480, 640, x_, K, K, R2, Tr)[2]
# #     Y.append(y)

# # plt.plot(x,Y)
# # plt.show()
# Q = triangulate_(640, 480, 320, 240, K1, K2, I, Tr)
# print(Q)
# result = DLT(camera_matrix1 @ np.hstack((R, t)),np.dot(camera_matrix2,np.hstack((R, t))),[100,300],[100,1])
# print(R)
# print(t)
# print(camera_matrix1 @ np.hstack((R, t)))
# print(np.hstack((R,t)))

# result = triangulate_points(0,0,100,100,camera_matrix1,dist_coeffs1,camera_matrix2,dist_coeffs2,R,t)
# print(result)
# result = triangulate_(200,100,0,200,camera_matrix1,camera_matrix2,R,t)
# print(np.linalg.norm(result))

# points1 = cv2.undistortPoints(np.array([[100, 200]], dtype=np.float32), K1, d, P=K1)[0][0]
# points1 = [points1[0],points1[1],1]
# pt1_norm = np.dot(np.linalg.inv(camera_matrix1), np.array([100, 200, 1]))

# print(points1)
# print(pt1_norm)