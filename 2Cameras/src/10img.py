import cv2
import os

capture = cv2.VideoCapture(2)

def prendre_photo(chemin_dossier, nom_photo):
    # Ouvrir la webcam
    
    
    # Lire l'image de la webcam
    while True:
        ret, frame = capture.read()
        
        # Afficher l'aperçu vidéo
        cv2.imshow("1", frame)
        
        # Attendre la touche "Entrée"
        if cv2.waitKey(1) == 13:  # 13 correspond à la touche "Entrée"
            break
    
    # Fermer la fenêtre de l'aperçu
    cv2.destroyAllWindows()
    
    # Vérifier si la capture a réussi
    if ret:
        # Créer le chemin complet du fichier de sortie
        chemin_photo = os.path.join(chemin_dossier, nom_photo)
        
        # Enregistrer l'image dans le dossier spécifié
        cv2.imwrite(chemin_photo, frame)
        print("La photo a été enregistrée avec succès : ", chemin_photo)
    else:
        print("Erreur lors de la capture de la photo.")


# Exemple d'utilisation
dossier_sortie = "C://Users//Abdellah.Koutit//Desktop//TestBench//2cameras//Calibration2"


# for i in range(10):

for i in range(10):

    nom_fichier = "camera"+str(i)+".jpg"

    prendre_photo(dossier_sortie, nom_fichier)