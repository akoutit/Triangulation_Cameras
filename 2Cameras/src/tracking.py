
from triangulate import *

from select_color import *
import time
import threading
from queue import Queue



Running = True


# Fonction pour détecter et suivre la couleur spécifiée
def track_color(i,frame, lower_color, upper_color):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Appliquer le masque pour détecter les pixels de la couleur spécifiée
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # Recherche des contours dans l'image
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    u, v = 0, 0
    # Trouver le plus grand contour
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(max_contour)

        # Vérifier si la taille du contour est suffisante pour être considérée comme l'objet
        if cv2.contourArea(max_contour) > 100:



            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)
            u, v = x, y

    # Afficher l'image
    cv2.imshow("frame"+str(i), frame)
    return u, v
def track1(output_queue):
    global Running
    video_capture = cv2.VideoCapture(0)
    
    
    while Running:
        # Lire un nouveau frame de la vidéo
        ret, frame = video_capture.read()
        
        if not ret:
            break

        
        # Suivre la couleur spécifiée
        u1, v1 = track_color(1,frame, lower_color1, upper_color1)
        track_color(1, frame, lower_color_yellow1, upper_color_yellow1)
        
        output_queue.put((u1,v1))
        


        # Sortir de la boucle si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):

            Running = False
        
    # Libérer les ressources
    video_capture.release()
    
    cv2.destroyAllWindows()
    
def track2(output_queue):

    global Running
    video_capture = cv2.VideoCapture(2)
    
    
    while Running:
        # Lire un nouveau frame de la vidéo
        ret, frame = video_capture.read()
        
        if not ret:
            break

        
        # Suivre la couleur spécifiée
        u2, v2 = track_color(2,frame, lower_color2, upper_color2)
       
        output_queue.put((u2,v2))

        


        # Sortir de la boucle si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('a'):
            Running = False
        
    # Libérer les ressources
    video_capture.release()
    
    cv2.destroyAllWindows()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def update_figure(ax, x, y, z):
    ax.cla()  # Efface la figure précédente
    ax.scatter(0,0,0)
    ax.scatter(x, y, z)  # Affiche le point en 3D
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Position du point en 3D')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])
    plt.pause(0.000001)      


def output(output_queue_1, output_queue_2, output_queue_3):

   

    global Running


    point = [0, 0, 0]  
    while   output_queue_1.empty() or output_queue_2.empty():
        # Création de la figure 3D
        
        
        u1,v1 = output_queue_1.get()
        u2,v2 = output_queue_2.get()
        if(u1 != 0 and u2 != 0):
            point = triangulate_points(u1, v1, u2, v2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, t )
            # point2 = triangulate_(u2, v2, u1, v1, camera_matrix2, dist_coeffs2, camera_matrix1, dist_coeffs1, R, t )
            # point3 = DLT(Proj2, Proj1, [u2, v2], [u1, v1])
            
            print("Triangulated Points      :             "  , point)
        output_queue_3.put(point)

        # Appel de la fonction pour mettre à jour la figure
        
        
            
                

            

        


def main():
    output_queue_1 = Queue()
    output_queue_2 = Queue()  
    output_queue_3 = Queue() 
    thread_1 = threading.Thread(target=track1, args=(output_queue_1,))
    thread_2 = threading.Thread(target=track2, args=(output_queue_2,))


    thread_3 = threading.Thread(target=output, args=(output_queue_1,output_queue_2,output_queue_3,))
    
    thread_1.start()
    thread_2.start()
    thread_3.start()
   


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Position du point en 3D')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])
    
    while len(output_queue_3.get())==3 :
        
        [x, y, z] = output_queue_3.get()
        
        update_figure(ax,-x,-z,-y)
    


    thread_1.join()
    thread_2.join()
    thread_3.join()


# Appeler la fonction principale
if __name__ == "__main__":
    main()
    
