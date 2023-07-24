import cv2
import numpy as np

def select_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, _ = frame.shape

    # Afficher le frame original et demander à l'utilisateur de sélectionner un rectangle autour de la couleur à suivre
    roi = cv2.selectROI("Frame", frame, False)

    # Convertir les coordonnées du rectangle sélectionné en plage de couleur HSV
    x, y, width, height = roi
    hsv_roi = hsv_frame[y:y+height, x:x+width]
    lower_color = np.min(hsv_roi, axis=(0, 1))
    upper_color = np.max(hsv_roi, axis=(0, 1))
    
    cv2.destroyWindow("Frame")

    
    return lower_color, upper_color

# Fonction principale
def main():
    # Ouvrir la capture vidéo depuis la webcam
    video_capture = cv2.VideoCapture(2)

    while True:
       

    # Lire le premier frame de la vidéo
        ret, frame = video_capture.read()
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 13:  # 13 correspond à la touche "Entrée"
            break
    # Sélectionner la couleur à suivre
    
    lower_color, upper_color = select_color(frame)
    






    video_capture.release()
    
    cv2.destroyAllWindows()
    print (lower_color, upper_color)
    return lower_color, upper_color 

lower_color_red1 = np.array([2, 204,  98])
upper_color_red1 = np.array([5, 255, 139])

lower_color_red2 = np.array([2, 156,  78])
upper_color_red2 = np.array([6, 255, 196])

lower_color_blue = np.array([100, 100, 100])
upper_color_blue = np.array([120, 255, 255])

lower_color_green = np.array([82, 192, 144])
upper_color_green = np.array([84, 229, 162])


lower_color_yellow1, upper_color_yellow1 = np.array([32,  93, 120]), np.array([37, 186, 168])
lower_color_yellow2, upper_color_yellow2 = np.array([22, 108, 166]), np.array([24, 163, 243])

lower_color1, upper_color1 = lower_color_red1, upper_color_red1
lower_color2, upper_color2 = lower_color_red2, upper_color_red2


# lower_color, upper_color = lower_color_green, upper_color_green
# lower_color_, upper_color_ = lower_color_green, upper_color_green

# lower_color, upper_color = lower_color_blue, upper_color_blue
# lower_color_, upper_color_ = lower_color_blue, upper_color_blue
# lower_color, upper_color = np.array([111, 183, 118]), np.array([117, 255, 247])
# lower_color_, upper_color_ = np.array([112, 213, 214]), np.array([113, 230, 255])

# lower_color, upper_color = main()



