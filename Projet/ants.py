"""
Module gérant une colonie de fourmis dans un labyrinthe.
"""
import numpy as np
import maze  # Importe le module de labyrinthe
import pheromone  # Importe le module de phéromones
import direction as d  # Importe le module de direction des fourmis
import pygame as pg  # Importe le module pygame pour les graphismes

UNLOADED, LOADED = False, True  # Définit des constantes pour le chargement/déchargement des fourmis

exploration_coefs = 0.  # Coefficients d'exploration initialisés à zéro


class Colony:
    """
    Représente une colonie de fourmis. Les fourmis ne sont pas individualisées pour des raisons de performance !
    """

    def __init__(self, nb_ants, pos_init, max_life):
        # Chaque fourmi a sa propre graine aléatoire unique
        self.seeds = np.arange(1, nb_ants + 1, dtype=np.int64)
        # État de chaque fourmi : chargée ou non chargée
        self.is_loaded = np.zeros(nb_ants, dtype=np.int8)
        # Calcul de la vie maximale pour chaque fourmi
        self.seeds[:] = np.mod(16807 * self.seeds[:], 2147483647)
        self.max_life = max_life * np.ones(nb_ants, dtype=np.int32)
        self.max_life -= np.int32(max_life * (self.seeds / 2147483647.)) // 4
        # Âge des fourmis : zéro au début
        self.age = np.zeros(nb_ants, dtype=np.int64)
        # Historique du chemin pris par chaque fourmi. La position à l'âge de la fourmi représente sa position actuelle.
        self.historic_path = np.zeros((nb_ants, max_life + 1, 2), dtype=np.int16)
        self.historic_path[:, 0, 0] = pos_init[0]
        self.historic_path[:, 0, 1] = pos_init[1]
        # Direction vers laquelle la fourmi est actuellement orientée (dépend de la direction d'où elle vient).
        self.directions = d.DIR_NONE * np.ones(nb_ants, dtype=np.int8)
        self.sprites = []  # Liste pour stocker les sprites d'animation des fourmis
        img = pg.image.load("ants.png").convert_alpha()  # Charge l'image des fourmis
        for i in range(0, 32, 8):
            self.sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))  # Découpe l'image en sprites individuels

    def return_to_nest(self, loaded_ants, pos_nest, food_counter):
        """
        Fonction qui ramène les fourmis transportant de la nourriture vers leur nid.
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
        """
        self.seeds[unloaded_ants] = np.mod(16807 * self.seeds[unloaded_ants], 2147483647)

        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0

        north_pos = np.copy(old_pos_ants)
        north_pos[:, 1] += 1
        north_pheromone = pheromones.pheromon[north_pos[:, 0], north_pos[:, 1]] * has_north_exit

        east_pos = np.copy(old_pos_ants)
        east_pos[:, 0] += 1
        east_pos[:, 1] += 2
        east_pheromone = pheromones.pheromon[east_pos[:, 0], east_pos[:, 1]] * has_east_exit

        south_pos = np.copy(old_pos_ants)
        south_pos[:, 0] += 2
        south_pos[:, 1] += 1
        south_pheromone = pheromones.pheromon[south_pos[:, 0], south_pos[:, 1]] * has_south_exit

        west_pos = np.copy(old_pos_ants)
        west_pos[:, 0] += 1
        west_pheromone = pheromones.pheromon[west_pos[:, 0], west_pos[:, 1]] * has_west_exit

        max_pheromones = np.maximum(north_pheromone, east_pheromone)
        max_pheromones = np.maximum(max_pheromones, south_pheromone)
        max_pheromones = np.maximum(max_pheromones, west_pheromone)

        choices = self.seeds[:] / 2147483647.

        ind_exploring_ants = np.nonzero(
            np.logical_or(choices[unloaded_ants] <= exploration_coefs, max_pheromones[unloaded_ants] == 0.))[0]
        if ind_exploring_ants.shape[0] > 0:
            ind_exploring_ants = unloaded_ants[ind_exploring_ants]
            valid_moves = np.zeros(choices.shape[0], np.int8)
            nb_exits = has_north_exit * np.ones(has_north_exit.shape) + has_east_exit * np.ones(has_east_exit.shape) + \
                       has_south_exit * np.ones(has_south_exit.shape) + has_west_exit * np.ones(has_west_exit.shape)
            while np.any(valid_moves[ind_exploring_ants] == 0):
                ind_ants_to_move = ind_exploring_ants[valid_moves[ind_exploring_ants] == 0]
                self.seeds[:] = np.mod(16807 * self.seeds[:], 2147483647)
                dir = np.mod(self.seeds[ind_ants_to_move], 4)
                old_pos = self.historic_path[ind_ants_to_move, self.age[ind_ants_to_move], :]
                new_pos = np.copy(old_pos)
                new_pos[:, 1] -= np.logical_and(dir == d.DIR_WEST,
                                                has_west_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0],
                                                                                               dtype=np.int16)
                new_pos[:, 1] += np.logical_and(dir == d.DIR_EAST,
                                                has_east_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0],
                                                                                               dtype=np.int16)
                new_pos[:, 0] -= np.logical_and(dir == d.DIR_NORTH,
                                                has_north_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0],
                                                                                               dtype=np.int16)
                new_pos[:, 0] += np.logical_and(dir == d.DIR_SOUTH,
                                                has_south_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0],
                                                                                               dtype=np.int16)
                valid_moves[ind_ants_to_move] = np.logical_or(new_pos[:, 0] != old_pos[:, 0],
                                                               new_pos[:, 1] != old_pos[:, 1])
                valid_moves[ind_ants_to_move] = np.logical_and(
                    valid_moves[ind_ants_to_move],
                    np.logical_or(dir != 3 - self.directions[ind_ants_to_move], nb_exits[ind_ants_to_move] == 1))
                ind_valid_moves = ind_ants_to_move[np.nonzero(valid_moves[ind_ants_to_move])[0]]
                self.historic_path[ind_valid_moves, self.age[ind_valid_moves] + 1, :] = new_pos[
                    valid_moves[ind_ants_to_move] == 1, :]
                self.directions[ind_valid_moves] = dir[valid_moves[ind_ants_to_move] == 1]

        ind_following_ants = np.nonzero(np.logical_and(choices[unloaded_ants] > exploration_coefs,
                                                       max_pheromones[unloaded_ants] > 0.))[0]
        if ind_following_ants.shape[0] > 0:
            ind_following_ants = unloaded_ants[ind_following_ants]
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, :] = \
                self.historic_path[ind_following_ants, self.age[ind_following_ants], :]
            max_east = (east_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, 1] += \
                max_east * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_west = (west_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, 1] -= \
                max_west * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_north = (north_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, 0] -= \
                max_north * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_south = (south_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, 0] += \
                max_south * np.ones(ind_following_ants.shape[0], dtype=np.int16)

        if unloaded_ants.shape[0] > 0:
            self.age[unloaded_ants] += 1

        ind_dying_ants = np.nonzero(self.age == self.max_life)[0]
        if ind_dying_ants.shape[0] > 0:
            self.age[ind_dying_ants] = 0
            self.historic_path[ind_dying_ants, 0, 0] = pos_nest[0]
            self.historic_path[ind_dying_ants, 0, 1] = pos_nest[1]
            self.directions[ind_dying_ants] = d.DIR_NONE

        ants_at_food_loc = np.nonzero(
            np.logical_and(self.historic_path[unloaded_ants, self.age[unloaded_ants], 0] == pos_food[0],
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

        [pheromones.mark(self.historic_path[i, self.age[i], :],
                         [has_north_exit[i], has_east_exit[i], has_west_exit[i], has_south_exit[i]]) for i in
         range(self.directions.shape[0])]
        return food_counter

    def display(self, screen):
        [screen.blit(self.sprites[self.directions[i]],
                     (8 * self.historic_path[i, self.age[i], 1], 8 * self.historic_path[i, self.age[i], 0])) for i
         in range(self.directions.shape[0])]


if __name__ == "__main__":
    import sys
    import time
    
    pg.init()
    # Récupère les paramètres de la ligne de commande
    size_laby = 25, 25
    # Si des paramètres sont passés en ligne de commande, les utilise
    if len(sys.argv) > 2:
        # Récupère la taille du labyrinthe
        size_laby = int(sys.argv[1]), int(sys.argv[2])
        # Si la taille est trop grande, la réduit

    resolution = size_laby[1] * 8, size_laby[0] * 8
    # Crée une fenêtre de la taille du labyrinthe
    screen = pg.display.set_mode(resolution)
    # Crée une colonie de fourmis
    nb_ants = size_laby[0] * size_laby[1] // 4
    # Si un nombre de fourmis est passé en ligne de commande, l'utilise
    max_life = 500
    # Si une durée de vie maximale est passée en ligne de commande, l'utilise
    if len(sys.argv) > 3:
        # Récupère le nombre de fourmis
        max_life = int(sys.argv[3])
        # Si la durée de vie est trop grande, la réduit
    pos_food = size_laby[0] - 1, size_laby[1] - 1
    # Positionne la nourriture dans le coin opposé au nid
    pos_nest = 0, 0
    # Positionne le nid dans le coin opposé à la nourriture
    a_maze = maze.Maze(size_laby, 12345)
    # Crée un labyrinthe
    ants = Colony(nb_ants, pos_nest, max_life)
    # Crée une colonie de fourmis
    unloaded_ants = np.array(range(nb_ants))
    # Crée un tableau de fourmis non chargées
    alpha = 0.9
    # Coefficient d'importance de la phéromone
    beta = 0.99
    # Coefficient d'importance de la distance
    if len(sys.argv) > 4:
        # Récupère les coefficients d'importance
        alpha = float(sys.argv[4])
        # Si le coefficient d'importance de la phéromone est trop grand, le réduit
    if len(sys.argv) > 5:
        # Récupère les coefficients d'importance
        beta = float(sys.argv[5])
        # Si le coefficient d'importance de la distance est trop grand, le réduit
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    # Crée un tableau de phéromones
    mazeImg = a_maze.display()
    # Crée une image du labyrinthe
    food_counter = 0
    # Compteur de nourriture

    snapshop_taken = False
    # Indique si une capture d'écran a déjà été prise
    while True:
        for event in pg.event.get():
            # Si l'utilisateur ferme la fenêtre, quitte le programme
            if event.type == pg.QUIT:
                # Ferme la fenêtre
                pg.quit()
                # Quitte le programme
                exit(0)

        deb = time.time()
        # Affiche les éléments à l'écran
        pherom.display(screen)
        # Affiche les phéromones
        screen.blit(mazeImg, (0, 0))
        # Affiche le labyrinthe
        ants.display(screen)
        # Affiche les fourmis
        pg.display.update()
        # Met à jour l'affichage

        food_counter = ants.advance(a_maze, pos_food, pos_nest, pherom, food_counter)
        # Fait avancer les fourmis
        pherom.do_evaporation(pos_food)
        # Fait évaporer les phéromones
        end = time.time()
        # Affiche les informations de débogage
        if food_counter == 1 and not snapshop_taken:
            # Prend une capture d'écran
            pg.image.save(screen, "MyFirstFood.png")
            # Indique qu'une capture d'écran a été prise
            snapshop_taken = True
            # Affiche les informations de débogage
        print(f"FPS : {1. / (end - deb):6.2f}, nourriture : {food_counter:7d}", end='\r')
        # Attend un peu pour ne pas surcharger le processeur

