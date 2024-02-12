# Importation des bibliothèques nécessaires
import numpy as np  
# Pour les opérations sur les tableaux
import matplotlib.pyplot as plt  
# Pour la création de graphiques
import multiprocessing  
# Pour le traitement multi-processus
from PIL import Image  
# Pour la manipulation des images
from time import time  
# Pour mesurer le temps d'exécution

# Définition de la classe MandelbrotSet 
# pour calculer l'ensemble de Mandelbrot
class MandelbrotSet:
    def __init__(self, max_iterations=50, escape_radius=2.0):
        # Initialisation des paramètres 
        # de l'ensemble de Mandelbrot
        self.max_iterations = max_iterations  
        # Nombre maximal d'itérations
        self.escape_radius = escape_radius   
        # Rayon d'échappement

    # Méthode pour calculer la convergence 
    # d'un point complexe sur l'ensemble de Mandelbrot
    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        # Calcul de la convergence du point c
        value = self.count_iterations(c, smooth) / self.max_iterations
        # Assurer que la valeur de convergence 
        # reste entre 0 et 1 si clamp est True
        return max(0.0, min(value, 1.0)) if clamp else value

    # Méthode pour compter les itérations 
    # nécessaires pour qu'un point complexe diverge
    def count_iterations(self, c: complex, smooth=False) -> int or float:
        # Initialisation des variables
        z = 0
        # Boucle pour effectuer les itérations
        for iter in range(self.max_iterations):
            z = z*z + c
            # Vérifier si le point a divergé
            if abs(z) > self.escape_radius:
                # Retourner le nombre d'itérations ou 
                # une valeur lissée si smooth est True
                if smooth:
                    return iter + 1 - np.log(np.log(abs(z))) / np.log(2)
                return iter
        return self.max_iterations

# Fonction pour calculer une ligne 
# de l'ensemble de Mandelbrot
def calculate_row(y):
    global mandelbrot_set, width, scaleX, scaleY
    row = np.empty(width)
    # Boucle pour calculer chaque pixel de la ligne y
    for x in range(width):
        # Calcul du point complexe correspondant au pixel (x, y)
        c = complex(-2. + scaleX * x, -1.125 + scaleY * y)
        # Calcul de la convergence du 
        # point c et stockage dans le tableau row
        row[x] = mandelbrot_set.convergence(c, smooth=True)
    return row

# Point d'entrée du programme
if __name__ == '__main__':
    # Initialisation des paramètres pour 
    # le calcul de l'ensemble de Mandelbrot
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024
    scaleX = 3. / width
    scaleY = 2.25 / height

    # Liste pour stocker les temps d'exécution 
    # pour différents nombres de processus
    times = []
    # Boucle pour tester différents nombres de processus
    for num_processes in range(1, multiprocessing.cpu_count() + 1):
        # Mesurer le temps de début d'exécution
        deb = time()
        # Créer un pool de processus avec 
        # le nombre spécifié de processus
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Mapper la fonction calculate_row 
            # sur les lignes de l'image
            results = pool.map(calculate_row, range(height))
        # Convertir les résultats en un tableau numpy
        convergence = np.array(results)
        # Mesurer le temps de fin d'exécution
        fin = time()
        # Calculer le temps d'exécution
        execution_time = fin - deb
        # Ajouter le temps d'exécution à la liste des temps
        times.append(execution_time)
        # Afficher le temps d'exécution 
        # pour le nombre de processus actuel
        print(f"Temps du calcul de l'ensemble de Mandelbrot avec {num_processes} processus : {execution_time}")

    # Calculer le temps d'exécution pour un seul processus
    single_process_time = times[0]
    # Calculer le speedup pour 
    # chaque configuration de processus
    speedups = [single_process_time / time for time in times]

    # Afficher le tableau des temps 
    # d'exécution et des speedups
    print("Nombre Processus - Temps du Calcul [s] - Speedup [S(n)]")
    for num_processes, execution_time, speedup in zip(range(1, multiprocessing.cpu_count() + 1), times, speedups):
        print(f"{num_processes} - {execution_time:.4f} - {speedup:.4f}")

    # Tracer le graphique du speedup 
    # en fonction du nombre de processus
    plt.subplot(2, 1, 1)
    plt.plot(range(1, multiprocessing.cpu_count() + 1), speedups, marker='o')
    plt.xlabel('Nombre de Processus')
    plt.ylabel('Speedup [S(n)]')
    plt.title('Speedup en fonction du Nombre de Processus')
    plt.grid(True)

    # Tracer le graphique du temps de calcul 
    # en fonction du nombre de processus
    plt.subplot(2, 1, 2)
    plt.plot(range(1, multiprocessing.cpu_count() + 1)
             , times, marker='o', color='r')
    plt.xlabel('Nombre de Processus')
    plt.ylabel('Temps de Calcul [s]')
    plt.title('Temps de Calcul en fonction du Nombre de Processus')
    plt.grid(True)

    # Ajuster les paramètres de disposition
    plt.tight_layout()
    # Afficher les graphiques
    plt.show()

    # Rassembler l'image et la sauvegarder
    image = np.vstack(results)
    image = (image / np.max(image)) * 255
    Image.fromarray(image.astype(np.uint8)).save('mandelbrot.png')
