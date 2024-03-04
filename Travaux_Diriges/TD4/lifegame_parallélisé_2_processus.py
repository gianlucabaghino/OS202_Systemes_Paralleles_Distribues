"""
TD4 - Parallélisation du jeu de la vie

1. Dans un premier temps, séparer l'affichage de la grille (et des cellules) du calcul des générations
(le code ne devra marcher que sur deux processus). Il est préférable que le processus 0 s'occupe de 
l'affichage et le 1 du calcul des générations

SOLUTION Q1 => Voici comment le code a été réorganisé pour séparer l'affichage de la grille du calcul des générations :

    Processus 0 (Affichage):
        Ce processus utilise Pygame pour afficher la grille du jeu de la vie.
        Il reçoit les données de la grille du processus 1 via MPI.
        Il est responsable de l'affichage continu de la grille mise à jour.

    Processus 1 (Calcul des générations):
        Ce processus est chargé du calcul des générations successives du jeu de la vie.
        Une fois qu'une génération est calculée, il envoie les données de la grille au processus 0 pour l'affichage.

La communication entre les processus se fait à l'aide de la bibliothèque MPI. 
Le processus 0 et le processus 1 sont synchronisés pour permettre un fonctionnement fluide du jeu de la vie.

Le code a été structuré de manière à ce que chaque processus exécute sa tâche spécifique de manière indépendante, 
contribuant ainsi à une séparation claire des responsabilités.
"""

'''
gianluca@gianluca-ASUS-TUF-Gaming-A15-FA506QM-FA506QM:~/Documents/OS202/TravauxDiriges/TD4$ mpiexec -n 2 python3 lifegame_parallélisé_2_processus.py glider 800 600
pygame 2.5.2 (SDL 2.28.2, Python 3.10.12)
Hello from the pygame community. https://www.pygame.org/contribute.html
pygame 2.5.2 (SDL 2.28.2, Python 3.10.12)
Hello from the pygame community. https://www.pygame.org/contribute.html
Pattern initial choisi : glider
resolution ecran : (800, 600)
Pattern initial choisi : glider
resolution ecran : (800, 600)
Generation  Computation Time    Drawing Time
0           7.38e-02            2.21e-02    
Generation  Computation Time    Drawing Time
1           7.54e-02            2.11e-02    
Generation  Computation Time    Drawing Time
2           7.27e-02            2.04e-02    
Generation  Computation Time    Drawing Time
3           7.25e-02            2.01e-02    
Generation  Computation Time    Drawing Time
4           7.22e-02            2.01e-02    
Generation  Computation Time    Drawing Time
5           7.30e-02            2.00e-02    
Generation  Computation Time    Drawing Time
6           7.26e-02            2.08e-02    
Generation  Computation Time    Drawing Time
7           7.27e-02            1.89e-02    
Generation  Computation Time    Drawing Time
8           7.30e-02            1.93e-02    
Generation  Computation Time    Drawing Time
9           7.27e-02            2.00e-02    
Generation  Computation Time    Drawing Time
10          7.26e-02            1.91e-02    
Generation  Computation Time    Drawing Time
11          7.32e-02            1.90e-02    
Generation  Computation Time    Drawing Time
12          7.26e-02            1.91e-02    
Generation  Computation Time    Drawing Time
13          7.35e-02            1.97e-02    
Generation  Computation Time    Drawing Time
14          7.29e-02            1.90e-02    
Generation  Computation Time    Drawing Time
15          7.34e-02            1.98e-02    
Generation  Computation Time    Drawing Time
16          7.30e-02            1.89e-02    
Generation  Computation Time    Drawing Time
17          7.32e-02            1.92e-02    
Generation  Computation Time    Drawing Time
18          7.31e-02            1.93e-02    
Generation  Computation Time    Drawing Time
19          7.38e-02            1.93e-02    
Generation  Computation Time    Drawing Time
20          7.32e-02            1.93e-02    
Generation  Computation Time    Drawing Time
21          7.33e-02            1.90e-02    
Generation  Computation Time    Drawing Time
22          7.32e-02            1.91e-02    
Generation  Computation Time    Drawing Time
23          7.37e-02            1.89e-02    
Generation  Computation Time    Drawing Time
24          7.29e-02            1.92e-02    
Generation  Computation Time    Drawing Time
25          7.34e-02            1.89e-02    
Generation  Computation Time    Drawing Time
26          7.28e-02            1.94e-02    
Generation  Computation Time    Drawing Time
27          7.35e-02            2.01e-02    
Generation  Computation Time    Drawing Time
28          7.35e-02            2.19e-02    
Generation  Computation Time    Drawing Time
29          7.41e-02            2.25e-02    
Generation  Computation Time    Drawing Time
30          7.50e-02            2.15e-02  
'''

import pygame as pg  # Importation de la bibliothèque Pygame pour la gestion des graphismes
import numpy as np   # Importation de la bibliothèque NumPy pour le calcul matriciel

# Définition de la classe Grille pour représenter la grille du jeu de la vie
class Grille:
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random
        self.dimensions = dim
        # Initialisation aléatoire des cellules si aucun schéma n'est fourni
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            # Extraction des indices pour le schéma initial
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            # Initialisation des cellules avec les indices correspondants
            self.cells[indices_i, indices_j] = 1
        else:
            # Initialisation aléatoire des cellules si aucun schéma n'est fourni
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life  # Couleur des cellules vivantes
        self.col_dead = color_dead  # Couleur des cellules mortes

    # Méthode pour calculer la prochaine itération du jeu de la vie
    def compute_next_iteration(self):
        ny = self.dimensions[0]  # Nombre de lignes
        nx = self.dimensions[1]  # Nombre de colonnes
        next_cells = np.empty(self.dimensions, dtype=np.uint8)  # Initialisation de la matrice pour la prochaine itération
        diff_cells = []  # Initialisation de la liste pour stocker les indices des cellules modifiées

        # Parcours de chaque cellule de la grille
        for i in range(ny):
            i_above = (i + ny - 1) % ny
            i_below = (i + 1) % ny
            for j in range(nx):
                j_left = (j - 1 + nx) % nx
                j_right = (j + 1) % nx
                # Indices des cellules voisines
                voisins_i = [i_above, i_above, i_above, i, i, i_below, i_below, i_below]
                voisins_j = [j_left, j, j_right, j_left, j_right, j_left, j, j_right]
                # Extraction des valeurs des cellules voisines
                voisines = np.array(self.cells[voisins_i, voisins_j])
                # Calcul du nombre de voisines vivantes
                nb_voisines_vivantes = np.sum(voisines)
                # Application des règles du jeu de la vie
                if self.cells[i, j] == 1:  # Si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[i, j] = 0  # Sous-population ou surpopulation, la cellule meurt
                        diff_cells.append(i * nx + j)
                    else:
                        next_cells[i, j] = 1  # Sinon elle reste vivante
                elif nb_voisines_vivantes == 3:  # Si cellule morte entourée exactement de trois vivantes
                    next_cells[i, j] = 1  # Naissance de la cellule
                    diff_cells.append(i * nx + j)
                else:
                    next_cells[i, j] = 0  # Morte, elle reste morte
        self.cells = next_cells  # Met à jour les cellules
        return diff_cells  # Retourne les indices des cellules modifiées

# Définition de la classe App pour l'interface graphique
class App:
    def __init__(self, geometry, grid):
        self.grid = grid
        # Calcul de la taille des cellules par rapport à la taille de la fenêtre et de la grille à afficher
        self.size_x = geometry[1] // grid.dimensions[1]
        self.size_y = geometry[0] // grid.dimensions[0]
        # Détermination de la couleur de la grille en fonction de sa taille
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None
        # Ajustement de la taille de la fenêtre pour bien ajuster la dimension de la grille
        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        # Création de la fenêtre Pygame
        self.screen = pg.display.set_mode((self.width, self.height))
        self.canvas_cells = []  # Initialisation de la liste des cellules

    # Méthode pour calculer le rectangle de chaque cellule
    def compute_rectangle(self, i: int, j: int):
        return (self.size_x * j, self.height - self.size_y * i - 1, self.size_x, self.size_y)

    # Méthode pour déterminer la couleur de chaque cellule en fonction de son état
    def compute_color(self, i: int, j: int):
        if self.grid.cells[i, j] == 0:
            return self.grid.col_dead
        else:
            return self.grid.col_life

    # Méthode pour dessiner la grille et les cellules
    def draw(self):
        # Remplissage de la grille avec les couleurs des cellules
        [self.screen.fill(self.compute_color(i, j), self.compute_rectangle(i, j)) for i in range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]
        # Dessin des lignes de la grille si nécessaire
        if self.draw_color is not None:
            [pg.draw.line(self.screen, self.draw_color, (0, i * self.size_y), (self.width, i * self.size_y)) for i in range(self.grid.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j * self.size_x, 0), (j * self.size_x, self.height)) for j in range(self.grid.dimensions[1])]
        pg.display.update()  # Mise à jour de l'affichage



if __name__ == '__main__':
    import time  # Importation du module time pour la gestion du temps
    import sys   # Importation du module sys pour interagir avec le système
    from mpi4py import MPI  # Importation du module MPI de mpi4py pour la communication parallèle
    import pickle  # Importation du module pickle pour la sérialisation des données

    if not MPI.Is_initialized():  # Vérification de l'initialisation de MPI
        MPI.Init()  # Initialisation de MPI

    comm = MPI.COMM_WORLD  # Création de l'objet de communication MPI
    rank = comm.Get_rank()  # Récupération du rang du processus MPI
    comm.Barrier()  # Synchronisation de tous les processus MPI

    size = comm.Get_size()  # Récupération du nombre de processus MPI

    if size != 2:  # Vérification du nombre de processus MPI nécessaire
        if rank == 0:
            print("This code requires exactly 2 MPI processes.")  # Affichage d'un message d'erreur si le nombre de processus est incorrect
        sys.exit(1)  # Sortie du programme avec un code d'erreur

    # Définition des schémas de grille disponibles avec leurs dimensions et configurations
    dico_patterns = {
        'blinker': ((5, 5), [(2, 1), (2, 2), (2, 3)]),
        'toad': ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
        "acorn": ((100, 100), [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)]),
        "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
        "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
        "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
        "glider_gun": ((200, 100), [(51, 76), (52, 74), (52, 76), (53, 64), (53, 65), (53, 72), (53, 73), (53, 86), (53, 87), (54, 63), (54, 67), (54, 72), (54, 73), (54, 86), (54, 87), (55, 52), (55, 53), (55, 62), (55, 68), (55, 72), (55, 73), (56, 52), (56, 53), (56, 62), (56, 66), (56, 68), (56, 69), (56, 74), (56, 76), (57, 62), (57, 68), (57, 76), (58, 63), (58, 67), (59, 64), (59, 65)]),
        "space_ship": ((25, 25), [(11, 13), (11, 14), (12, 11), (12, 12), (12, 14), (12, 15), (13, 11), (13, 12), (13, 13), (13, 14), (14, 12), (14, 13)]),
        "die_hard": ((100, 100), [(51, 57), (52, 51), (52, 52), (53, 52), (53, 56), (53, 57), (53, 58)]),
        "pulsar": ((17, 17), [(2, 4), (2, 5), (2, 6), (7, 4), (7, 5), (7, 6), (9, 4), (9, 5), (9, 6), (14, 4), (14, 5), (14, 6), (2, 10), (2, 11), (2, 12), (7, 10), (7, 11), (7, 12), (9, 10), (9, 11), (9, 12), (14, 10), (14, 11), (14, 12), (4, 2), (5, 2), (6, 2), (4, 7), (5, 7), (6, 7), (4, 9), (5, 9), (6, 9), (4, 14), (5, 14), (6, 14), (10, 2), (11, 2), (12, 2), (10, 7), (11, 7), (12, 7), (10, 9), (11, 9), (12, 9), (10, 14), (11, 14), (12, 14)]),
        "floraison": ((40, 40), [(19, 18), (19, 19), (19, 20), (20, 17), (20, 19), (20, 21), (21, 18), (21, 19), (21, 20)]),
        "block_switch_engine": ((400, 400), [(201, 202), (201, 203), (202, 202), (202, 203), (211, 203), (212, 204), (212, 202), (214, 204), (214, 201), (215, 201), (215, 202), (216, 201)]),
        "u": ((200, 200), [(101, 101), (102, 102), (103, 102), (103, 101), (104, 103), (105, 103), (105, 102), (105, 101), (105, 105), (103, 105), (102, 105), (101, 105), (101, 104)]),
        "flat": ((200, 400), [(80, 200), (81, 200), (82, 200), (83, 200), (84, 200), (85, 200), (86, 200), (87, 200), (89, 200), (90, 200), (91, 200), (92, 200), (93, 200), (97, 200), (98, 200), (99, 200), (106, 200), (107, 200), (108, 200), (109, 200), (110, 200), (111, 200), (112, 200), (114, 200), (115, 200), (116, 200), (117, 200), (118, 200)])
    }
    
    choice = 'glider'  # Choix par défaut du schéma de grille
    if len(sys.argv) > 1:  # Vérification de la présence d'un argument pour le choix du schéma de grille
        choice = sys.argv[1]

    resx = 800  # Définition de la résolution par défaut en X
    resy = 800  # Définition de la résolution par défaut en Y
    if len(sys.argv) > 3:  # Vérification de la présence d'arguments pour la résolution de l'écran
        resx = int(sys.argv[2])  # Résolution en X fournie en argument
        resy = int(sys.argv[3])  # Résolution en Y fournie en argument
    print(f"Pattern initial choisi : {choice}")  # Affichage du schéma de grille choisi
    print(f"resolution ecran : {resx, resy}")  # Affichage de la résolution de l'écran

    pg.init()  # Initialisation de Pygame

    start_time = time.time()  # Enregistrement du temps de début d'exécution du programme
    generation = 0  # Initialisation du compteur de générations

    try:
        if rank == 1:  # Si le processus est le numéro 1
            try:
                init_pattern = dico_patterns[choice]  # Sélection du schéma de grille initial
            except KeyError:
                print("No such pattern. Available ones are:", dico_patterns.keys())  # Affichage d'un message d'erreur si le schéma de grille n'existe pas
                exit(1)  # Sortie du programme avec un code d'erreur

            grid = Grille(*init_pattern)  # Création de la grille dans le processus 1
            comm.send(grid.cells, dest=0)  # Envoi de la matrice de cellules au processus 0

            while True:  # Boucle infinie pour le calcul des itérations suivantes
                t1 = time.time()  # Enregistrement du temps de début de calcul
                grid.compute_next_iteration()  # Calcul de la prochaine itération de la grille
                t2 = time.time()  # Enregistrement du temps de fin de calcul
                comm.send(grid.cells, dest=0)  # Envoi de la matrice de cellules au processus 0
                sys.stdout.flush()  # Vidage du tampon de sortie standard

        elif rank == 0:  # Si le processus est le numéro 0
            cells = comm.recv(source=1)  # Réception de la matrice de cellules du processus 1
            grid = Grille(cells.shape, None, pg.Color("black"), pg.Color("white"))  # Création de la grille avec les cellules reçues
            appli = App((resx, resy), grid)  # Création de l'application avec la grille reçue

            while True:  # Boucle infinie pour la réception et l'affichage des itérations
                cells = comm.recv(source=1)  # Réception de la matrice de cellules du processus 1
                grid.cells = cells  # Mise à jour de la matrice de cellules de la grille locale
                appli.draw()  # Dessin de la grille
                
                t1 = time.time()  # Enregistrement du temps de début de calcul
                diff = grid.compute_next_iteration()  # Calcul de la prochaine itération de la grille
                t2 = time.time()  # Enregistrement du temps de fin de calcul
                appli.draw()  # Dessin de la grille
                t3 = time.time()  # Enregistrement du temps de fin de dessin
                print(f"{'Generation':<12}{'Computation Time':<20}{'Drawing Time':<12}")  # Affichage de l'en-tête du tableau
                print(f"{generation:<12}{t2 - t1:<20.2e}{t3 - t2:<12.2e}")  # Affichage des temps de calcul et de dessin
                generation += 1  # Incrémentation du compteur de générations

                for event in pg.event.get():  # Boucle pour la gestion des événements Pygame
                    if event.type == pg.QUIT:  # Si l'événement est de quitter
                        pg.quit()  # Fermeture de Pygame
                        comm.Abort(0)  # Arrêt des processus MPI
                        sys.exit()  # Sortie du programme

    except KeyboardInterrupt:  # Gestion de l'interruption du programme
        print("\nEnd of program")  # Affichage de la fin du programme
        end_time = time.time()  # Enregistrement du temps de fin d'exécution du programme
        total_time = end_time - start_time  # Calcul du temps total d'exécution
        print(f"Total time: {total_time}")  # Affichage du temps total d'exécution

