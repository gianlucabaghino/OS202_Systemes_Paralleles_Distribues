"""
Parallélisation du code : 
Séparant affichage (sur le proc 0) et gestion des fourmis/phéromones (sur le proc 1)
"""

import numpy as np  # Importation de la bibliothèque NumPy pour les calculs numériques
import maze  # Importation du module "maze" pour la gestion du labyrinthe
import pheromone  # Importation du module "pheromone" pour la gestion des phéromones
import direction as d  # Importation du module "direction" pour la gestion des directions
import pygame as pg  # Importation de la bibliothèque Pygame pour la visualisation graphique

UNLOADED, LOADED = False, True  # Définition des constantes pour l'état des fourmis (non chargées et chargées)

exploration_coefs = 0.  # Coefficients d'exploration initialisés à zéro

class Colony:
    """
    Représente une colonie de fourmis. Les fourmis ne sont pas individualisées pour des raisons de performance !

    Entrées:
        nb_ants : Nombre de fourmis dans la fourmilière
        pos_init : Positions initiales des fourmis (position de la fourmilière)
        max_life : Durée de vie maximale que les fourmis peuvent atteindre
    """
    def __init__(self, nb_ants, pos_init, max_life):
        # Chaque fourmi a sa propre graine aléatoire unique
        self.seeds = np.arange(1, nb_ants+1, dtype=np.int64)
        # État de chaque fourmi : chargée ou non chargée
        self.is_loaded = np.zeros(nb_ants, dtype=np.int8)
        # Calcul de la durée de vie maximale pour chaque fourmi :
        #   Mise à jour de la graine aléatoire :
        self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
        # Durée de vie de chaque fourmi = 75% à 100% de la durée de vie maximale des fourmis
        self.max_life = max_life * np.ones(nb_ants, dtype=np.int32)
        self.max_life -= np.int32(max_life*(self.seeds/2147483647.))//4
        # Âges des fourmis : zéro au début
        self.age = np.zeros(nb_ants, dtype=np.int64)
        # Historique du chemin pris par chaque fourmi. La position à l'âge de la fourmi représente sa position actuelle.
        self.historic_path = np.zeros((nb_ants, max_life+1, 2), dtype=np.int16)
        self.historic_path[:, 0, 0] = pos_init[0]
        self.historic_path[:, 0, 1] = pos_init[1]
        # Direction vers laquelle la fourmi est actuellement orientée (dépend de la direction d'où elle vient).
        self.directions = d.DIR_NONE*np.ones(nb_ants, dtype=np.int8)
        self.sprites = []  # Initialisation de la liste des sprites pour les fourmis
        img = pg.image.load("ants.png").convert_alpha()  # Chargement de l'image des fourmis
        for i in range(0, 32, 8):
            self.sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))

    def return_to_nest(self, loaded_ants, pos_nest, food_counter):
        """
        Fonction qui ramène les fourmis transportant de la nourriture à leur nid.

        Entrées:
            loaded_ants : Indices des fourmis transportant de la nourriture
            pos_nest : Position du nid où les fourmis doivent aller
            food_counter : Quantité actuelle de nourriture dans le nid

        Renvoie la nouvelle quantité de nourriture
        """
        self.age[loaded_ants] -= 1

        in_nest_tmp = self.historic_path[loaded_ants, self.age[loaded_ants], :] == pos_nest
        if in_nest_tmp.any():
            in_nest_loc = np.nonzero(np.logical_and(in_nest_tmp[:, 0], in_nest_tmp[:, 1]))[0]
            if in_nest_loc.shape[0] > 0:
                in_nest = loaded_ants[in_nest_loc]
                self.is_loaded[in_nest] = UNLOADED
                self.age[in_nest] = 0
                food_counter += in_nest_loc.shape[0]
        return food_counter

    def explore(self, unloaded_ants, the_maze, pos_food, pos_nest, pheromones):
        """
        Gestion des fourmis non chargées explorant le labyrinthe.

        Entrées :
            unloaded_ants : Indices des fourmis non chargées
            the_maze : Le labyrinthe dans lequel les fourmis se déplacent
            pos_food : Position de la nourriture dans le labyrinthe
            pos_nest : Position du nid des fourmis dans le labyrinthe
            pheromones : La carte des phéromones (qui a également des cellules fantômes pour une gestion plus facile des bords)

        Sorties : Aucune
        """
        # Mise à jour de la graine aléatoire (pour pseudo-aléatoire manuel) appliquée à toutes les fourmis non chargées
        self.seeds[unloaded_ants] = np.mod(16807*self.seeds[unloaded_ants], 2147483647)

        # Calcul des sorties possibles pour chaque fourmi dans le labyrinthe :
        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0

        # Lecture des phéromones voisines :
        north_pos = np.copy(old_pos_ants)
        north_pos[:, 1] += 1
        north_pheromone = pheromones.pheromon[north_pos[:, 0], north_pos[:, 1]]*has_north_exit

        east_pos = np.copy(old_pos_ants)
        east_pos[:, 0] += 1
        east_pos[:, 1] += 2
        east_pheromone = pheromones.pheromon[east_pos[:, 0], east_pos[:, 1]]*has_east_exit

        south_pos = np.copy(old_pos_ants)
        south_pos[:, 0] += 2
        south_pos[:, 1] += 1
        south_pheromone = pheromones.pheromon[south_pos[:, 0], south_pos[:, 1]]*has_south_exit

        west_pos = np.copy(old_pos_ants)
        west_pos[:, 0] += 1
        west_pheromone = pheromones.pheromon[west_pos[:, 0], west_pos[:, 1]]*has_west_exit

        max_pheromones = np.maximum(north_pheromone, east_pheromone)
        max_pheromones = np.maximum(max_pheromones, south_pheromone)
        max_pheromones = np.maximum(max_pheromones, west_pheromone)

        # Calcul des choix pour toutes les fourmis non chargées (pour les autres, nous calculons mais cela n'a pas d'importance)
        choices = self.seeds[:] / 2147483647.

        # Les fourmis explorent le labyrinthe par choix ou si aucune phéromone ne peut les guider :
        ind_exploring_ants = np.nonzero(
            np.logical_or(choices[unloaded_ants] <= exploration_coefs, max_pheromones[unloaded_ants] == 0.))[0]
        if ind_exploring_ants.shape[0] > 0:
            ind_exploring_ants = unloaded_ants[ind_exploring_ants]
            valid_moves = np.zeros(choices.shape[0], np.int8)
            nb_exits = has_north_exit * np.ones(has_north_exit.shape) + has_east_exit * np.ones(has_east_exit.shape) + \
                has_south_exit * np.ones(has_south_exit.shape) + has_west_exit * np.ones(has_west_exit.shape)
            while np.any(valid_moves[ind_exploring_ants] == 0):
                # Calcul des indices des fourmis dont le dernier mouvement n'était pas valide :
                ind_ants_to_move = ind_exploring_ants[valid_moves[ind_exploring_ants] == 0]
                self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
                # Choix d'une direction aléatoire :
                dir = np.mod(self.seeds[ind_ants_to_move], 4)
                old_pos = self.historic_path[ind_ants_to_move, self.age[ind_ants_to_move], :]
                new_pos = np.copy(old_pos)
                new_pos[:, 1] -= np.logical_and(dir == d.DIR_WEST,
                                                has_west_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 1] += np.logical_and(dir == d.DIR_EAST,
                                                has_east_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] -= np.logical_and(dir == d.DIR_NORTH,
                                                has_north_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] += np.logical_and(dir == d.DIR_SOUTH,
                                                has_south_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                # Mouvement valide si nous ne sommes pas restés en place à cause d'un mur
                valid_moves[ind_ants_to_move] = np.logical_or(new_pos[:, 0] != old_pos[:, 0], new_pos[:, 1] != old_pos[:, 1])
                # et si nous ne sommes pas dans la direction opposée du mouvement précédent (et s'il y a d'autres sorties)
                valid_moves[ind_ants_to_move] = np.logical_and(
                    valid_moves[ind_ants_to_move],
                    np.logical_or(dir != 3-self.directions[ind_ants_to_move], nb_exits[ind_ants_to_move] == 1))
                # Calcul des indices des fourmis dont le mouvement vient d'être validé :
                ind_valid_moves = ind_ants_to_move[np.nonzero(valid_moves[ind_ants_to_move])[0]]
                # Pour ces fourmis, nous mettons à jour leurs positions et directions
                self.historic_path[ind_valid_moves, self.age[ind_valid_moves] + 1, :] = new_pos[valid_moves[ind_ants_to_move] == 1, :]
                self.directions[ind_valid_moves] = dir[valid_moves[ind_ants_to_move] == 1]

        ind_following_ants = np.nonzero(np.logical_and(choices[unloaded_ants] > exploration_coefs,
                                                       max_pheromones[unloaded_ants] > 0.))[0]
        if ind_following_ants.shape[0] > 0:
            ind_following_ants = unloaded_ants[ind_following_ants]
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, :] = \
                self.historic_path[ind_following_ants, self.age[ind_following_ants], :]
            max_east = (east_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] += \
                max_east * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_west = (west_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] -= \
                max_west * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_north = (north_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] -= max_north * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_south = (south_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] += max_south * np.ones(ind_following_ants.shape[0], dtype=np.int16)

        # Vieillissement d'une unité pour l'âge des fourmis non chargées
        if unloaded_ants.shape[0] > 0:
            self.age[unloaded_ants] += 1

        # Élimination des fourmis en fin de vie :
        ind_dying_ants = np.nonzero(self.age == self.max_life)[0]
        if ind_dying_ants.shape[0] > 0:
            self.age[ind_dying_ants] = 0
            self.historic_path[ind_dying_ants, 0, 0] = pos_nest[0]
            self.historic_path[ind_dying_ants, 0, 1] = pos_nest[1]
            self.directions[ind_dying_ants] = d.DIR_NONE

        # Pour les fourmis atteignant la nourriture, nous mettons à jour leurs états :
        ants_at_food_loc = np.nonzero(np.logical_and(self.historic_path[unloaded_ants, self.age[unloaded_ants], 0] == pos_food[0],
                                                     self.historic_path[unloaded_ants, self.age[unloaded_ants], 1] == pos_food[1]))[0]
        if ants_at_food_loc.shape[0] > 0:
            ants_at_food = unloaded_ants[ants_at_food_loc]
            self.is_loaded[ants_at_food] = True

    def advance(self, the_maze, pos_food, pos_nest, pheromones, food_counter=0):
        loaded_ants = np.nonzero(self.is_loaded == True)[0]
        unloaded_ants = np.nonzero(self.is_loaded == False)[0]
        if loaded_ants.shape[0] > 0:
            food_counter = self.return_to_nest(loaded_ants, pos_nest, food_counter)
        if unloaded_ants.shape[0] > 0:
            self.explore(unloaded_ants, the_maze, pos_food, pos_nest, pheromones)

        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0
        # Marquage des phéromones :
        [pheromones.mark(self.historic_path[i, self.age[i], :],
                         [has_north_exit[i], has_east_exit[i], has_west_exit[i], has_south_exit[i]]) for i in range(self.directions.shape[0])]
        return food_counter

    def display(self, screen):
        [screen.blit(self.sprites[self.directions[i]], (8*self.historic_path[i, self.age[i], 1], 8*self.historic_path[i, self.age[i], 0])) for i in range(self.directions.shape[0])]


if __name__ == "__main__":
    import sys
    import time
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    # Récupérer le rang et la taille du communicateur
    rank = comm.Get_rank() 
    size = comm.Get_size()

    pg.init()
    # Paramètres par défaut
    size_laby = 25, 25
    # Si des paramètres sont passés en ligne de commande, les utiliser
    if len(sys.argv) > 2:
        # Récupérer la taille du labyrinthe
        size_laby = int(sys.argv[1]),int(sys.argv[2])
        # Si la taille est trop grande, la réduire

    resolution = size_laby[1]*8, size_laby[0]*8
    # Créer une fenêtre graphique
    screen = pg.display.set_mode(resolution)
    # Initialiser les paramètres de la simulation
    nb_ants = size_laby[0]*size_laby[1]//4
    # Si le nombre de fourmis est trop grand, le réduire
    max_life = 500
    # Si la durée de vie maximale est trop grande, la réduire
    if len(sys.argv) > 3:
        # Récupérer le nombre de fourmis
        max_life = int(sys.argv[3])
        # Si la durée de vie maximale est trop grande, la réduire
    pos_food = size_laby[0]-1, size_laby[1]-1
    # Position de la nourriture
    pos_nest = 0, 0
    # Position de la fourmilière
    a_maze = maze.Maze(size_laby, 12345)
    # Créer un labyrinthe
    ants = Colony(nb_ants, pos_nest, max_life)
    # Créer une colonie de fourmis
    unloaded_ants = np.array(range(nb_ants))
    # Fourmis non chargées
    alpha = 0.9
    # Coefficient d'importance des phéromones
    beta  = 0.99
    # Coefficient d'importance de la visibilité
    if len(sys.argv) > 4:
        # Récupérer les coefficients d'importance
        alpha = float(sys.argv[4])
        # Coefficient d'importance des phéromones
    if len(sys.argv) > 5:
        # Récupérer les coefficients d'importance
        beta = float(sys.argv[5])
        # Coefficient d'importance de la visibilité
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    # Créer une carte de phéromones
    mazeImg = a_maze.display()
    # Afficher le labyrinthe
    food_counter = 0
    # Compteur de nourriture
    snapshop_taken = False
    # Indicateur de capture d'écran

    if rank == 1:
        # Code de gestion des fourmis et des phéromones
        while True:
            food_counter = ants.advance(a_maze, pos_food, pos_nest, pherom, food_counter)
            # Évolution des fourmis
            pherom.do_evaporation(pos_food)
            # Évaporation des phéromones
            end = time.time()
            ant_data = [ants.age, ants.is_loaded, ants.directions]
            # Recueillir les données de chaque fourmi
            last_positions = [ants.historic_path[i, ants.age[i], :] for i in range(ants.directions.shape[0])]
            # Rassembler la dernière position de chaque fourmi
            comm.send((pherom, food_counter, snapshop_taken, ant_data, last_positions), dest=0)
            # Envoyer les fourmis mises à jour, les phéromones, last_positions et ant_data à rang 0

    elif rank == 0:
        # Code pour l'affichage
        while True:
            for event in pg.event.get():
                # Gestion des événements
                if event.type == pg.QUIT:
                    # Si l'utilisateur ferme la fenêtre, quitter
                    pg.quit()
                    # Fermer la fenêtre
                    comm.Abort(0)
                    # Terminer le programme
                    sys.exit()

            deb = time.time()
            # Recevoir les fourmis mises à jour, les phéromones, last_positions et ant_data de rang 1
            pherom, food_counter, snapshop_taken, ant_data, last_positions = comm.recv(source=1)
            ants.age, ants.is_loaded, ants.directions = ant_data
            # Mettre à jour les fourmis
            # Ajouter les dernières positions à l'historique des positions
            for i, last_position in enumerate(last_positions):
                ants.historic_path[i, ants.age[i], :] = last_position
                # Mettre à jour les phéromones
            pherom.display(screen)
            # Afficher les phéromones
            screen.blit(mazeImg, (0, 0))
            # Afficher le labyrinthe
            ants.display(screen) # Afficher les fourmis
            pg.display.update() # Mettre à jour l'affichage

            if food_counter == 1 and not snapshop_taken:
                # Prendre un instantané de l'écran
                pg.image.save(screen, "MyFirstFood2.png")
                # Sauvegarder l'instantané
                snapshop_taken = True
            end = time.time()  # Définir 'end' ici
            print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}", flush=True)
            # Afficher le nombre de FPS et le nombre de fourmis transportant de la nourriture
