import numpy as np
import cv2
import glob
# Liste des chemins d'accès aux images capturées

num_camera = 2

image_paths = glob.glob('Calibration'+str(num_camera)+'/*.jpg')

# Taille d'un carré de la grille de chessboard en unités arbitraires
square_size = 1

# Nombre de coins internes de la grille de chessboard
chessboard_size = (8, 6)

# Préparer les points de la grille de chessboard
object_points = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
object_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
object_points *= square_size

# Variables pour stocker les points de coins détectés
image_points = []  # Points 2D dans les images (coins détectés)
object_points_list = []  # Points 3D dans l'espace réel (coordonnées des coins)

# Parcourir les images capturées
for image_path in image_paths:
    # Lire l'image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Impossible de lire l'image {image_path}")
        continue

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Trouver les coins de la grille de chessboard dans l'image
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Ajouter les points détectés à la liste
        image_points.append(corners)
        object_points_list.append(object_points)

        # Dessiner les coins détectés sur l'image
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

        # Afficher l'image avec les coins détectés
        cv2.imshow("Chessboard", image)
        cv2.waitKey(500)

# Fermer toutes les fenêtres
cv2.destroyAllWindows()

# Calibrer la caméra
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points_list, image_points, gray.shape[::-1], None, None
)

# Afficher les paramètres de calibration
print("Matrice de caméra (intrinsèque):")
print(camera_matrix)
np.save('camera_matrix'+str(num_camera)+'.npy', camera_matrix)
print("\nCoefficients de distorsion:")
print(dist_coeffs)
np.save('dist_coeffs'+str(num_camera)+'.npy', dist_coeffs)

