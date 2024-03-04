"""
Le jeu de la vie
################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonales.) L'interaction se fait selon les
règles suivantes pour calculer l'irération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

On itère ensuite pour étudier la façon dont évolue la population des cellules sur la grille.
"""

'''
gianluca@gianluca-ASUS-TUF-Gaming-A15-FA506QM-FA506QM:~/Documents/OS202/TravauxDiriges/TD4$ python3 lifegame.py glider 800 600
pygame 2.5.2 (SDL 2.28.2, Python 3.10.12)
Hello from the pygame community. https://www.pygame.org/contribute.html
Pattern initial choisi : glider
resolution ecran : (800, 600)
Génération  Temps de Calcul     Temps de Dessin
0           7.22e-02            2.21e-02    
Génération  Temps de Calcul     Temps de Dessin
1           7.47e-02            2.20e-02    
Génération  Temps de Calcul     Temps de Dessin
2           7.39e-02            2.12e-02    
Génération  Temps de Calcul     Temps de Dessin
3           7.20e-02            2.01e-02    
Génération  Temps de Calcul     Temps de Dessin
4           7.16e-02            2.00e-02    
Génération  Temps de Calcul     Temps de Dessin
5           7.19e-02            1.90e-02    
Génération  Temps de Calcul     Temps de Dessin
6           7.15e-02            1.89e-02    
Génération  Temps de Calcul     Temps de Dessin
7           7.13e-02            2.00e-02    
Génération  Temps de Calcul     Temps de Dessin
8           7.19e-02            2.00e-02    
Génération  Temps de Calcul     Temps de Dessin
9           7.13e-02            1.98e-02    
Génération  Temps de Calcul     Temps de Dessin
10          7.13e-02            2.18e-02    
Génération  Temps de Calcul     Temps de Dessin
11          7.24e-02            2.11e-02    
Génération  Temps de Calcul     Temps de Dessin
12          7.24e-02            1.93e-02    
Génération  Temps de Calcul     Temps de Dessin
13          7.16e-02            2.02e-02    
Génération  Temps de Calcul     Temps de Dessin
14          7.16e-02            1.97e-02    
Génération  Temps de Calcul     Temps de Dessin
15          7.12e-02            1.97e-02    
Génération  Temps de Calcul     Temps de Dessin
16          7.15e-02            1.99e-02    
Génération  Temps de Calcul     Temps de Dessin
17          7.13e-02            2.01e-02    
Génération  Temps de Calcul     Temps de Dessin
18          8.10e-02            2.26e-02    
Génération  Temps de Calcul     Temps de Dessin
19          7.64e-02            2.19e-02    
Génération  Temps de Calcul     Temps de Dessin
20          7.60e-02            2.11e-02    
Génération  Temps de Calcul     Temps de Dessin
21          7.24e-02            2.08e-02    
Génération  Temps de Calcul     Temps de Dessin
22          7.13e-02            2.03e-02    
Génération  Temps de Calcul     Temps de Dessin
23          7.37e-02            1.99e-02    
Génération  Temps de Calcul     Temps de Dessin
24          7.13e-02            1.96e-02    
Génération  Temps de Calcul     Temps de Dessin
25          7.13e-02            1.97e-02    
Génération  Temps de Calcul     Temps de Dessin
26          7.22e-02            2.00e-02    
Génération  Temps de Calcul     Temps de Dessin
27          7.78e-02            1.98e-02    
Génération  Temps de Calcul     Temps de Dessin
28          7.65e-02            1.97e-02    
Génération  Temps de Calcul     Temps de Dessin
29          7.12e-02            1.97e-02    
Génération  Temps de Calcul     Temps de Dessin
30          7.18e-02            1.97e-02   
'''

import pygame  as pg
# Importe le module pygame pour la création de l'interface graphique
import numpy   as np
# Importe le module numpy pour les opérations sur les matrices

class Grille:
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
    # Constructeur de la classe Grille
        import random
        # Importe le module random pour générer des nombres aléatoires
        self.dimensions = dim
        # Stocke les dimensions de la grille
        if init_pattern is not None: 
        # Si un pattern est donné, on initialise les cellules avec ce pattern
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            # Crée une matrice de zéros de la taille de la grille
            indices_i = [v[0] for v in init_pattern]
            # Récupère les indices i des cellules vivantes
            indices_j = [v[1] for v in init_pattern]
            # Récupère les indices j des cellules vivantes
            self.cells[indices_i,indices_j] = 1
            # Met à 1 les cellules vivantes
        else:
        # Sinon, on initialise les cellules aléatoirement
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
            # Remplit la grille de 0 et de 1 aléatoirement
        self.col_life = color_life
        # Stocke la couleur des cellules vivantes
        self.col_dead = color_dead
        # Stocke la couleur des cellules mortes

    def compute_next_iteration(self):
    # Méthode de la classe Grille pour calculer la prochaine itération du jeu de la vie
        # Remarque 1: 
        # on pourrait optimiser en faisant du vectoriel, 
        # mais pour plus de clarté, on utilise les boucles
        
        # Remarque 2: 
        # on voit la grille plus comme une matrice qu'une grille géométrique. 
        # L'indice (0,0) est donc en haut à gauche de la grille !
        ny = self.dimensions[0]
        # Nombre de lignes
        nx = self.dimensions[1]
        # Nombre de colonnes
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        # Crée une matrice vide de la taille de la grille
        diff_cells = []
        # Liste des cellules qui ont changé d'état
        for i in range(ny):
            # Pour chaque ligne
            i_above = (i+ny-1)%ny
            # Indice de la ligne au dessus
            i_below = (i+1)%ny
            # Indice de la ligne en dessous
            for j in range(nx):
                # Pour chaque colonne
                j_left = (j-1+nx)%nx
                # Indice de la colonne à gauche
                j_right= (j+1)%nx
                # Indice de la colonne à droite
                voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below]
                # Liste des indices i des cellules voisines
                voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right]
                # Liste des indices j des cellules voisines
                voisines = np.array(self.cells[voisins_i,voisins_j])
                # Liste des cellules voisines
                nb_voisines_vivantes = np.sum(voisines)
                # Nombre de cellules voisines vivantes
                if self.cells[i,j] == 1: 
                # Si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                    # Si la cellule a moins de deux voisines vivantes ou plus de trois
                        next_cells[i,j] = 0 
                        # Cas de sous ou sur population, la cellule meurt
                        diff_cells.append(i*nx+j)
                        # On ajoute la cellule à la liste des cellules qui ont changé d'état
                    else:
                    # Sinon elle reste vivante
                        next_cells[i,j] = 1 
                        # La cellule reste vivante
                elif nb_voisines_vivantes == 3: 
                # Cas où cellule morte mais entourée exactement de trois vivantes
                    next_cells[i,j] = 1        
                    # La cellule devient vivante
                    diff_cells.append(i*nx+j)
                    # On ajoute la cellule à la liste des cellules qui ont changé d'état
                else:
                    next_cells[i,j] = 0
                    # Sinon la cellule reste morte
        self.cells = next_cells
        # On met à jour la grille
        return diff_cells
        # On retourne la liste des cellules qui ont changé d'état


class App:
    def __init__(self, geometry, grid):
    # Constructeur de la classe App
    # Crée une fenêtre pour afficher la grille
        self.grid = grid
        # Stocke la grille
        # Calcul de la taille d'une cellule par rapport à la taille de la fenêtre et de la grille à afficher :
        self.size_x = geometry[1]//grid.dimensions[1]
        # Taille d'une cellule en largeur
        self.size_y = geometry[0]//grid.dimensions[0]
        # Taille d'une cellule en hauteur
        if self.size_x > 4 and self.size_y > 4 :
        # Si la taille d'une cellule est suffisante, on dessine les lignes de la grille
            self.draw_color=pg.Color('lightgrey')
            # Couleur des lignes de la grille
        else:
        # Sinon, on ne dessine pas les lignes de la grille
            self.draw_color=None
            # Pas de couleur pour les lignes de la grille
        # Ajustement de la taille de la fenêtre pour bien fitter la dimension de la grille
        self.width = grid.dimensions[1] * self.size_x
        # Calcul de la largeur de la fenêtre
        self.height= grid.dimensions[0] * self.size_y
        # Calcul de la hauteur de la fenêtre
        # Création de la fenêtre à l'aide de tkinter
        self.screen = pg.display.set_mode((self.width,self.height))
        # Crée une fenêtre de la taille de la grille
        self.canvas_cells = []
        # Liste des cellules à afficher

    def compute_rectangle(self, i: int, j: int):
    # Méthode de la classe App pour calculer la géométrie du rectangle correspondant à la cellule (i,j)
        return (self.size_x*j, self.height - self.size_y*i - 1, self.size_x, self.size_y)
        # Retourne les coordonnées du rectangle correspondant à la cellule (i,j)

    def compute_color(self, i: int, j: int):
    # Méthode de la classe App pour calculer la couleur de la cellule (i,j)
        if self.grid.cells[i,j] == 0:
        # Si la cellule est morte
            return self.grid.col_dead
            # On retourne la couleur des cellules mortes
        else:
            return self.grid.col_life
            # Sinon on retourne la couleur des cellules vivantes

    def draw(self):
    # Méthode de la classe App pour dessiner la grille
        [self.screen.fill(self.compute_color(i,j),self.compute_rectangle(i,j)) for i in range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]
        # Dessine les cellules de la grille
        if (self.draw_color is not None):
        # Si on doit dessiner les lignes de la grille
            [pg.draw.line(self.screen, self.draw_color, (0,i*self.size_y), (self.width,i*self.size_y)) for i in range(self.grid.dimensions[0])]
            # Dessine les lignes horizontales de la grille
            [pg.draw.line(self.screen, self.draw_color, (j*self.size_x,0), (j*self.size_x,self.height)) for j in range(self.grid.dimensions[1])]
            # Dessine les lignes verticales de la grille
        pg.display.update()
        # Met à jour l'affichage de la fenêtre


if __name__ == '__main__':
    import time
    # Importe le module time pour mesurer le temps
    import sys
    # Importe le module sys pour récupérer les arguments de la ligne de commande

    pg.init()
    # Initialise le module pygame
    dico_patterns = { # Dimension et pattern dans un tuple
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
    choice = 'glider'
    # Choix du pattern initial
    if len(sys.argv) > 1 :
    # Si un pattern est donné en argument
        choice = sys.argv[1]
        # On le récupère
    resx = 800
    #  Résolution par défaut
    resy = 800
    #  Résolution par défaut
    if len(sys.argv) > 3 :
    # Si une résolution est donnée en argument
        resx = int(sys.argv[2])
        # On la récupère
        resy = int(sys.argv[3])
        # On la récupère
    print(f"Pattern initial choisi : {choice}")
    # Affiche le pattern initial choisi
    print(f"resolution ecran : {resx,resy}")
    # Affiche la résolution de l'écran
    try:
    # On essaie de récupérer le pattern initial
        init_pattern = dico_patterns[choice]
        # On récupère le pattern initial
    except KeyError:
    # Si le pattern n'existe pas
        print("No such pattern. Available ones are:", dico_patterns.keys())
        # On affiche les patterns disponibles
        exit(1)
        # On quitte le programme
    grid = Grille(*init_pattern)
    #  Crée une grille avec le pattern initial
    appli = App((resx, resy), grid)
    # Crée une application avec la grille
    
    start_time = time.time()
    # Mesure le temps de début

    generation = 0
    # Initialisation du compteur de génération
    try:
    # On essaie de faire tourner le programme
        while True:
        # Tant que l'utilisateur n'arrête pas le programme
            t1 = time.time() 
            # Mesure le temps de calcul de la prochaine génération
            diff = grid.compute_next_iteration()
            # Calcule la prochaine génération
            t2 = time.time()
            # Mesure le temps de calcul de la prochaine génération
            appli.draw()
            # Dessine la grille
            t3 = time.time()
            # Mesure le temps d'affichage
            for event in pg.event.get():
            # Pour chaque événement
                if event.type == pg.QUIT:
                # Si l'événement est de quitter
                    pg.quit()
                    # On quitte pygame
            '''
            print(f"Temps calcul prochaine generation : {t2-t1:2.2e} secondes, temps affichage : {t3-t2:2.2e} secondes\r", end='');
            '''  
            print(f"{'Génération':<12}{'Temps de Calcul':<20}{'Temps de Dessin':<12}")   
            print(f"{generation:<12}{t2-t1:<20.2e}{t3-t2:<12.2e}")  # Print the generation number, computation time, and drawing time
            generation += 1 
            # Affiche le temps de calcul de la prochaine génération et le temps d'affichage
    except KeyboardInterrupt:
        # Si l'utilisateur appuie sur Ctrl+C
        print("\nFin du programme")
        end_time = time.time()
        # Mesure le temps de fin
        sequential_time = end_time - start_time
        # Mesure le temps total
        print(f"Temps total: {sequential_time}")
        
