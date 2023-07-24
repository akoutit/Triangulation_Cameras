import cv2
import numpy as np

# Créer un détecteur de points d'intérêt (SIFT ou autre)
detector = cv2.SIFT_create()

# Définir les paramètres du suivi optique
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def track_object(frame, prev_pts, prev_gray):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculer les nouveaux points à suivre avec le suivi optique
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    # Sélectionner les points valides et mettre à jour les points de suivi
    valid_pts = next_pts[status == 1]
    prev_pts = valid_pts.reshape(-1, 1, 2)

    # Dessiner les points de suivi sur l'image
    for pt in valid_pts:
        x, y = pt.ravel()
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Mettre à jour l'image et les points précédents pour la prochaine itération
    cv2.imshow("Frame", frame)
    prev_gray = gray.copy()

    return prev_pts, prev_gray

# Ouvrir la vidéo ou capturer les images à partir d'une source en direct
cap = cv2.VideoCapture(0)

# Lire la première image pour initialiser le suivi
ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Sélectionner un rectangle initial autour de l'objet à suivre
x, y, w, h = cv2.selectROI("Frame", frame, False)
prev_pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32).reshape(-1, 1, 2)

# Boucle principale de suivi
while True:
    # Lire l'image suivante
    ret, frame = cap.read()
    if not ret:
        break

    # Effectuer le suivi de l'objet
    prev_pts, prev_gray = track_object(frame, prev_pts, prev_gray)

    # Quitter la boucle en appuyant sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
