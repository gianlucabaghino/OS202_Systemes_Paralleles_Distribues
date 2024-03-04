"""
TD4 - Parallélisation du jeu de la vie

2. Puis en utilisant la méthode Split des objets de type communicateur, paralléliser le calcul des 
générations de sorte que le 0 continue à afficher les grilles et les cellules tandis que les processus a
llant de 1 à nbp-1 calculent en parallèle les nouvelles générations de cellules.

SOLUTION Q2 => Voici comment cette approche est implémentée dans le code :

    Initialisation de la communication MPI :
        Tout d'abord, la bibliothèque MPI est importée (from mpi4py import MPI).
        Ensuite, le communicateur initial comm0 est créé en utilisant MPI.COMM_WORLD.
        La taille du groupe de communication est obtenue avec size = comm0.Get_size().
        Le rang dans le groupe de communication est obtenu avec rank = comm0.Get_rank().

    Division du communicateur :
        Le processus maître (0) divise le communicateur en deux groupes en utilisant la méthode Split. 
        Le paramètre color=1 est utilisé pour différencier les deux groupes.
        La division est effectuée comme suit : comm = comm0.Split(color=1)

    Logique de traitement parallèle :
        Si le processus est le maître (0), il continue à afficher les grilles et les cellules dans une boucle principale, et il gère les événements Pygame.
        Si le processus est différent du maître (c'est-à-dire les processus allant de 1 à size - 1), il reçoit les cellules à mettre à jour depuis le processus 0, 
        calcule la prochaine itération pour sa partie de la grille, puis envoie les cellules mises à jour au processus 0.
        Le processus maître reçoit ensuite les cellules mises à jour de tous les autres processus et met à jour sa propre grille en conséquence.

    Gestion de la boucle principale :
        Le processus maître gère une boucle principale dans laquelle il calcule et affiche les nouvelles générations de cellules à intervalles réguliers.
        Pendant ce temps, les autres processus (1 à size - 1) reçoivent les cellules à mettre à jour, 
        calculent la prochaine itération et envoient les résultats au processus maître.
"""

'''
gianluca@gianluca-ASUS-TUF-Gaming-A15-FA506QM-FA506QM:~/Documents/OS202/TravauxDiriges/TD4$ mpiexec -n 4 python3 lifegame_parallélisé_plusieurs_processus.py glider 800 600
pygame 2.5.2 (SDL 2.28.2, Python 3.10.12)
Hello from the pygame community. https://www.pygame.org/contribute.html
pygame 2.5.2 (SDL 2.28.2, Python 3.10.12)
Hello from the pygame community. https://www.pygame.org/contribute.html
pygame 2.5.2 (SDL 2.28.2, Python 3.10.12)
Hello from the pygame community. https://www.pygame.org/contribute.html
pygame 2.5.2 (SDL 2.28.2, Python 3.10.12)
Hello from the pygame community. https://www.pygame.org/contribute.html
Pattern initial choisi : glider
resolution ecran : (800, 600)
35
36
Pattern initial choisi : glider
resolution ecran : (800, 600)
35
36
Pattern initial choisi : glider
resolution ecran : (800, 600)
35
36
Pattern initial choisi : glider
resolution ecran : (800, 600)
35
36
Génération  Temps de Calcul     Temps de Dessin
0           8.56e-02            2.99e-02    
Génération  Temps de Calcul     Temps de Dessin
1           7.99e-02            1.99e-02    
Génération  Temps de Calcul     Temps de Dessin
2           7.90e-02            1.97e-02    
Génération  Temps de Calcul     Temps de Dessin
3           7.87e-02            1.99e-02    
Génération  Temps de Calcul     Temps de Dessin
4           7.75e-02            1.93e-02    
Génération  Temps de Calcul     Temps de Dessin
5           7.77e-02            1.95e-02    
Génération  Temps de Calcul     Temps de Dessin
6           7.84e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
7           7.87e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
8           8.25e-02            1.95e-02    
Génération  Temps de Calcul     Temps de Dessin
9           7.76e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
10          7.76e-02            1.94e-02    
Génération  Temps de Calcul     Temps de Dessin
11          7.75e-02            1.94e-02    
Génération  Temps de Calcul     Temps de Dessin
12          7.80e-02            1.94e-02    
Génération  Temps de Calcul     Temps de Dessin
13          7.81e-02            1.95e-02    
Génération  Temps de Calcul     Temps de Dessin
14          7.91e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
15          7.87e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
16          7.88e-02            1.98e-02    
Génération  Temps de Calcul     Temps de Dessin
17          7.84e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
18          7.82e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
19          7.84e-02            1.97e-02    
Génération  Temps de Calcul     Temps de Dessin
20          7.80e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
21          7.82e-02            1.95e-02    
Génération  Temps de Calcul     Temps de Dessin
22          7.81e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
23          7.86e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
24          7.85e-02            1.98e-02    
Génération  Temps de Calcul     Temps de Dessin
25          7.78e-02            1.94e-02    
Génération  Temps de Calcul     Temps de Dessin
26          7.81e-02            1.97e-02    
Génération  Temps de Calcul     Temps de Dessin
27          8.49e-02            2.21e-02    
Génération  Temps de Calcul     Temps de Dessin
28          8.74e-02            2.12e-02    
Génération  Temps de Calcul     Temps de Dessin
29          8.29e-02            2.14e-02    
Génération  Temps de Calcul     Temps de Dessin
30          8.19e-02            2.15e-02    
'''

import pygame as pg  # Importation de la bibliothèque Pygame pour la gestion des graphiques
import numpy as np   # Importation de la bibliothèque NumPy pour le calcul numérique
import time          # Importation de la bibliothèque time pour la gestion du temps
import sys           # Importation de la bibliothèque sys pour les interactions système
from mpi4py import MPI  # Importation de la bibliothèque MPI pour le calcul distribué

class Grille:
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random  # Importation du module random
        self.dimensions = dim  # Initialisation des dimensions de la grille
        if init_pattern is not None:  # Vérification si un motif initial est fourni
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)  # Initialisation des cellules avec des zéros
            indices_i = [v[0] for v in init_pattern]  # Extraction des indices i
            indices_j = [v[1] for v in init_pattern]  # Extraction des indices j
            self.cells[indices_i, indices_j] = 1  # Initialisation des cellules vivantes avec des uns
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)  # Initialisation aléatoire des cellules
        self.col_life = color_life  # Couleur des cellules vivantes
        self.col_dead = color_dead  # Couleur des cellules mortes

    def compute_next_iteration(self):
        ny = self.dimensions[0]  # Nombre de lignes
        nx = self.dimensions[1]  # Nombre de colonnes
        next_cells = np.empty(self.dimensions, dtype=np.uint8)  # Initialisation des prochaines cellules
        diff_cells = []  # Initialisation de la liste des cellules différentes
        for i in range(ny):  # Parcours des lignes
            i_above = (i + ny - 1) % ny  # Indice de la ligne au-dessus
            i_below = (i + 1) % ny  # Indice de la ligne en dessous
            for j in range(nx):  # Parcours des colonnes
                j_left = (j - 1 + nx) % nx  # Indice de la colonne à gauche
                j_right = (j + 1) % nx  # Indice de la colonne à droite
                voisins_i = [i_above, i_above, i_above, i, i, i_below, i_below, i_below]  # Indices i des voisins
                voisins_j = [j_left, j, j_right, j_left, j_right, j_left, j, j_right]  # Indices j des voisins
                voisines = np.array(self.cells[voisins_i, voisins_j])  # Récupération des cellules voisines
                nb_voisines_vivantes = np.sum(voisines)  # Calcul du nombre de voisins vivants
                if self.cells[i, j] == 1:  # Si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):  # Si sous-population ou surpopulation
                        next_cells[i, j] = 0  # La cellule meurt
                        diff_cells.append(i * nx + j)  # Ajout de la cellule différente
                    else:
                        next_cells[i, j] = 1  # La cellule reste vivante
                elif nb_voisines_vivantes == 3:  # Si la cellule est morte et a exactement 3 voisins vivants
                    next_cells[i, j] = 1  # Naissance de la cellule
                    diff_cells.append(i * nx + j)  # Ajout de la cellule différente
                else:
                    next_cells[i, j] = 0  # La cellule reste morte
        self.cells = next_cells  # Mise à jour des cellules
        return diff_cells  # Retourne les cellules différentes

class App:
    def __init__(self, geometry, grid):
        self.grid = grid  # Initialisation de la grille
        # Calcul de la taille d'une cellule par rapport à la taille de la fenêtre et de la grille à afficher :
        self.size_x = geometry[1] // grid.dimensions[1]  # Taille horizontale d'une cellule
        self.size_y = geometry[0] // grid.dimensions[0]  # Taille verticale d'une cellule
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')  # Couleur de dessin de la grille
        else:
            self.draw_color = None
        # Ajustement de la taille de la fenêtre pour bien fitter la dimension de la grille
        self.width = grid.dimensions[1] * self.size_x  # Largeur de la fenêtre
        self.height = grid.dimensions[0] * self.size_y  # Hauteur de la fenêtre
        # Création de la fenêtre à l'aide de tkinter
        self.screen = pg.display.set_mode((self.width, self.height))  # Initialisation de la fenêtre
        self.canvas_cells = []  # Initialisation du canevas des cellules

    def compute_rectangle(self, i: int, j: int):
        return (self.size_x * j, self.height - self.size_y * i - 1, self.size_x, self.size_y)  # Calcul du rectangle

    def compute_color(self, i: int, j: int):
        if self.grid.cells[i, j] == 0:
            return self.grid.col_dead  # Retourne la couleur des cellules mortes
        else:
            return self.grid.col_life  # Retourne la couleur des cellules vivantes

    def draw(self):
        # Dessin des cellules
        [self.screen.fill(self.compute_color(i, j), self.compute_rectangle(i, j)) for i in
         range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]
        if self.draw_color is not None:
            # Dessin des lignes de la grille si la couleur est spécifiée
            [pg.draw.line(self.screen, self.draw_color, (0, i * self.size_y), (self.width, i * self.size_y)) for i in
             range(self.grid.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j * self.size_x, 0), (j * self.size_x, self.height)) for j in
             range(self.grid.dimensions[1])]
        pg.display.update()  # Mise à jour de l'affichage


if __name__ == '__main__':
    import tkinter as tk  # Importation de tkinter pour l'interface graphique
    import time            # Importation de la bibliothèque time pour la gestion du temps
    import numpy as np     # Importation de la bibliothèque NumPy pour le calcul numérique
    import pygame as pg    # Importation de Pygame pour la création de jeux et d'applications multimédias
    from mpi4py import MPI  # Importation de la bibliothèque MPI pour le calcul distribué
    import sys             # Importation de la bibliothèque sys pour les interactions système

    dico_patterns = { # Dictionnaire contenant des modèles de jeu de la vie
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((200,100),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }

    choice = 'glider'  # Choix par défaut du modèle de jeu de la vie

    if len(sys.argv) > 1 :  # Si des arguments sont passés en ligne de commande
        choice = sys.argv[1]  # Utilisez le premier argument comme choix de modèle

    resx = 800  # Résolution par défaut en largeur
    resy = 800  # Résolution par défaut en hauteur

    if len(sys.argv) > 3 :  # Si plus de trois arguments sont passés en ligne de commande
        resx = int(sys.argv[2])  # Utilisez le deuxième argument comme résolution en largeur
        resy = int(sys.argv[3])  # Utilisez le troisième argument comme résolution en hauteur

    print(f"Pattern initial choisi : {choice}")  # Affichage du modèle initial choisi
    print(f"resolution ecran : {resx,resy}")     # Affichage de la résolution de l'écran

    init_pattern = dico_patterns[choice]  # Sélection du modèle initial à partir du dictionnaire

    comm0 = MPI.COMM_WORLD  # Initialisation de la communication MPI
    size = comm0.Get_size()  # Obtention de la taille du groupe de communication
    rank = comm0.Get_rank()  # Obtention du rang dans le groupe de communication

    ny, nx = dico_patterns[choice][0]  # Obtention des dimensions du modèle choisi
    div = ny // (size - 1)  # Calcul de la division des lignes entre les threads
    print(div + 2)  # Affichage du nombre de lignes pour chaque thread
    print(ny - (size - 2) * div + 2)  # Affichage du reste de lignes pour le dernier thread

    comm = comm0.Split(color=1)  # Division du communicateur en deux groupes

    if rank == 0:  # Si le processus est le maître
        grid = Grille(*init_pattern)  # Initialisation de la grille avec le modèle choisi
        appli = App((resx,resy),grid)  # Initialisation de l'application graphique

        start_time = time.time()  # Enregistrement du temps de début

        generation = 0  # Initialisation du compteur de génération

        try:
            while True:  # Boucle principale
                t1 = time.time()  # Temps avant le calcul de la prochaine génération
                diff = grid.compute_next_iteration()  # Calcul de la prochaine génération
                t2 = time.time()  # Temps après le calcul de la prochaine génération
                appli.draw()  # Affichage de la grille
                t3 = time.time()  # Temps après l'affichage
                for event in pg.event.get():  # Gestion des événements Pygame
                    if event.type == pg.QUIT:  # Si l'utilisateur quitte
                        pg.quit()  # Fermeture de Pygame
                print(f"{'Génération':<12}{'Temps de Calcul':<20}{'Temps de Dessin':<12}")
                print(f"{generation:<12}{t2-t1:<20.2e}{t3-t2:<12.2e}")  # Affichage du numéro de génération et des temps de calcul et de dessin
                generation += 1  # Incrémentation du compteur de génération

        except KeyboardInterrupt:  # En cas d'interruption par l'utilisateur
            print("\nFin du programme")  # Message de fin
            end_time = time.time()  # Temps de fin
            sequential_time = end_time - start_time  # Calcul du temps total
            print(f"Temps total: {sequential_time}")  # Affichage du temps total

    elif rank == size - 1:  # Si le processus est le dernier
        dim_x_last_thread = ny - (size - 2) * div + 2  # Calcul de la dimension en x du dernier thread
        partial_grid = Grille(dim=(dim_x_last_thread,nx))  # Initialisation de la grille partielle

    else:  # Si le processus est un thread intermédiaire
        partial_grid = Grille(dim=(div+2,nx))  # Initialisation de la grille partielle

    if rank == 0:  # Si le processus est le maître
        while True:  # Boucle principale
            for k in range(1, size):  # Pour chaque processus
                if k == 1:
                    cells = np.vstack((grid.cells[-1:], grid.cells[0:div+1,:]))
                    comm.send(cells, dest=k)
                elif k == size - 1:
                    cells = np.vstack((grid.cells[(k - 1) * div - 1:,:], grid.cells[0,:]))
                    comm.send(cells, dest=k)
                else:
                    cells = grid.cells[(k - 1) * div - 1:k * div + 1,:]
                    comm.send(cells, dest=k)

            cell = []

            nx = grid.dimensions[1]  # Obtention de la dimension x de la grille
            ny = grid.dimensions[0]  # Obtention de la dimension y de la grille
            for k in range(1, size):  # Pour chaque processus MPI
                cells_new = comm0.recv(source=k)  # Réception des cellules mises à jour du processus k
                if k == 1:
                    grid.cells[0:div,:] = cells_new  # Mise à jour des cellules dans la première tranche de la grille
                elif k == size - 1:
                    grid.cells[(k-1)*div:,:] = cells_new  # Mise à jour des cellules dans la dernière tranche de la grille
                else:
                    grid.cells[(k-1)*div:k*div,:] = cells_new  # Mise à jour des cellules dans les tranches intermédiaires
                        
            appli.draw()  # Dessin de la grille mise à jour
                
            for event in pg.event.get():  # Pour chaque événement dans la boucle d'événements Pygame
                if event.type == pg.QUIT:  # Si l'événement est de quitter la fenêtre
                    pg.quit()  # Fermeture de la fenêtre Pygame
                    comm.Abort(0)  # Arrêt de la communication MPI
                    comm0.Abort(0)  # Arrêt de la communication MPI
                    sys.exit()  # Fermeture du programme

    else:  # Pour les autres processus MPI
        while True:  # Boucle infinie pour continuer à recevoir et envoyer les cellules
            cells = comm.recv(source=0)  # Réception des cellules à mettre à jour depuis le processus 0
            partial_grid.cells = cells  # Mise à jour des cellules de la grille partielle avec les données reçues
            partial_grid.compute_next_iteration()  # Calcul de la prochaine itération pour la grille partielle
            comm0.send(partial_grid.cells[1:-1, :], dest=0)  # Envoi des cellules mises à jour au processus 0



