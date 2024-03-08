"""
Parallélisation du code : 
En partitionnant les fourmis entre les processus dont le rang est non nul (le zéro continuant à gérer l'affichage)
"""

from mpi4py import MPI  # Importer la bibliothèque MPI pour la communication entre processus
import numpy as np  # Importer numpy pour les opérations numériques
import maze  # Importer le module maze pour la création de labyrinthes
import pheromone  # Importer le module pheromone pour la gestion des phéromones
import direction as d  # Importer le module direction pour les directions des fourmis
import pygame as pg  # Importer pygame pour l'affichage graphique
import sys  # Importer sys pour accéder aux arguments de la ligne de commande
import time  # Importer time pour la gestion du temps

comm = MPI.COMM_WORLD  # Initialiser la communication MPI
rank = comm.Get_rank()  # Obtenir le rang du processus actuel
size = comm.Get_size()  # Obtenir le nombre total de processus

UNLOADED, LOADED = False, True  # Définir des constantes pour le chargement des fourmis
exploration_coefs = 0.  # Initialiser le coefficient d'exploration

# Classe Colony pour représenter la colonie de fourmis
class Colony:
    def __init__(self, nb_ants, pos_init, max_life):
        # Initialisation des paramètres des fourmis
        self.seeds = np.arange(1, nb_ants + 1, dtype=np.int64)
        self.is_loaded = np.zeros(nb_ants, dtype=np.int8)
        self.seeds[:] = np.mod(16807 * self.seeds[:], 2147483647)
        self.max_life = max_life * np.ones(nb_ants, dtype=np.int32)
        self.max_life -= np.int32(max_life * (self.seeds / 2147483647.)) // 4
        self.age = np.zeros(nb_ants, dtype=np.int64)
        self.historic_path = np.zeros((nb_ants, max_life + 1, 2), dtype=np.int16)
        self.historic_path[:, 0, 0] = pos_init[0]
        self.historic_path[:, 0, 1] = pos_init[1]
        self.directions = d.DIR_NONE * np.ones(nb_ants, dtype=np.int8)
        self.sprites = []
        img = pg.image.load("ants.png").convert_alpha()
        for i in range(0, 32, 8):
            self.sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))

    # Méthode pour le retour au nid des fourmis chargées
    def return_to_nest(self, loaded_ants, pos_nest, food_counter):
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

    # Méthode pour l'exploration des fourmis non chargées
    def explore(self, unloaded_ants, the_maze, pos_food, pos_nest, pheromones):
        self.seeds[unloaded_ants] = np.mod(16807 * self.seeds[unloaded_ants], 2147483647)
        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0

        # Calcul des positions et phéromones dans toutes les directions
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

        # Exploration des fourmis en fonction des phéromones et des choix aléatoires
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

        # Suivi des phéromones pour les fourmis restantes
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
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, 0] -= max_north * np.ones(
                ind_following_ants.shape[0], dtype=np.int16)
            max_south = (south_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, 0] += max_south * np.ones(
                ind_following_ants.shape[0], dtype=np.int16)

        # Incrémenter l'âge des fourmis non chargées
        if unloaded_ants.shape[0] > 0:
            self.age[unloaded_ants] += 1

        # Gérer les fourmis mourantes et celles atteignant la nourriture
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

    # Méthode pour faire avancer les fourmis dans le labyrinthe
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

    # Méthode pour afficher les fourmis sur l'écran
    def display(self, screen):
        [screen.blit(self.sprites[self.directions[i]],
                      (8 * self.historic_path[i, self.age[i], 1], 8 * self.historic_path[i, self.age[i], 0])) for i
         in range(self.directions.shape[0])]

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    # Initialiser la communication MPI
    rank = comm.Get_rank()
    # Obtenir le rang du processus actuel
    size = comm.Get_size()
    # Obtenir le nombre total de processus
    
    pg.init()
    # Initialiser pygame
    size_laby = 25, 25
    # Initialiser la taille du labyrinthe
    if len(sys.argv) > 2:
        # Si des arguments sont passés en ligne de commande
        size_laby = int(sys.argv[1]), int(sys.argv[2])
        # Utiliser ces arguments pour la taille du labyrinthe
    resolution = size_laby[1] * 8, size_laby[0] * 8
    # Initialiser la résolution de l'écran
    screen = pg.display.set_mode(resolution)
    # Initialiser l'écran
    nb_ants = size_laby[0] * size_laby[1] // 4
    # Initialiser le nombre de fourmis
    max_life = 500
    # Initialiser la durée de vie maximale des fourmis
    if len(sys.argv) > 3:
        # Si un troisième argument est passé en ligne de commande
        max_life = int(sys.argv[3])
        # Utiliser cet argument pour la durée de vie maximale des fourmis
    pos_food = size_laby[0] - 1, size_laby[1] - 1
    # Initialiser la position de la nourriture
    pos_nest = 0, 0
    # Initialiser la position du nid
    ants = Colony(nb_ants, pos_nest, max_life)
    # Initialiser la colonie de fourmis
    a_maze = maze.Maze(size_laby, 12345)
    # Initialiser le labyrinthe
    unloaded_ants = np.array(range(nb_ants))
    # Initialiser les fourmis non chargées
    alpha = 0.9
    # Initialiser le coefficient alpha
    beta = 0.99
    # Initialiser le coefficient beta
    if len(sys.argv) > 4:
        # Si un quatrième argument est passé en ligne de commande
        alpha = float(sys.argv[4])
        # Utiliser cet argument pour le coefficient alpha
    if len(sys.argv) > 5:
        # Si un cinquième argument est passé en ligne de commande
        beta = float(sys.argv[5])
        # Utiliser cet argument pour le coefficient beta
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    # Initialiser les phéromones
    mazeImg = a_maze.display()
    # Initialiser l'image du labyrinthe
    food_counter = 0
    # Initialiser le compteur de nourriture
    snapshop_taken = False
    # Initialiser le booléen pour la capture d'écran

    nb_ants_l = nb_ants//(size-1)
    # Initialiser le nombre de fourmis par processus
    nb_remainder = nb_ants%(size-1)
    # Initialiser le reste de la division du nombre de fourmis par le nombre de processus
    
    if rank == size-1:
        # Si le processus actuel est le dernier
        ants_l = Colony(nb_ants_l+nb_remainder, pos_nest, max_life)
        # Initialiser la colonie de fourmis
    elif rank > 0:
        # Si le processus actuel n'est pas le premier
        ants_l = Colony(nb_ants_l, pos_nest, max_life)
        # Initialiser la colonie de fourmis

    if rank > 0:
        # Si le processus actuel n'est pas le premier
        while True:
            food_counter = ants_l.advance(a_maze, pos_food, pos_nest, pherom, food_counter)
            # Faire avancer les fourmis
            pherom.do_evaporation(pos_food)
            # Faire évaporer les phéromones
            end = time.time()
            # Initialiser la fin du temps
            ant_data = np.array([ants_l.age, ants_l.is_loaded, ants_l.directions])
            # Initialiser les données des fourmis
            last_positions = np.array([ants_l.historic_path[i, ants_l.age[i], :] for i in range(ants_l.directions.shape[0])])
            # Initialiser les dernières positions des fourmis
            comm.send((pherom, food_counter, snapshop_taken, ant_data, last_positions), dest=0)
            # Envoyer les données au processus 0

    if rank == 0:
        # Si le processus actuel est le premier
        while True:
            for event in pg.event.get():
                # Pour chaque événement
                if event.type == pg.QUIT:
                    # Si l'événement est de quitter
                    pg.quit()
                    # Quitter pygame
                    comm.Abort(0)
                    sys.exit()

            deb = time.time()
            # Initialiser le début du temps
            age_l = []
            # Initialiser la liste des âges
            loaded_l = []
            # Initialiser la liste des chargements
            direction_l = []
            # Initialiser la liste des directions
            last_positions = []
            # Initialiser la liste des dernières positions

            for source_rank in range(1, size):
                # Pour chaque processus
                pherom, food_counter_l, snapshop_taken, ant_data_l, last_positions_l = comm.recv(source=source_rank)
                # Recevoir les données du processus
                age_l.append(ant_data_l[0])
                # Ajouter les âges à la liste
                loaded_l.append(ant_data_l[1])
                # Ajouter les chargements à la liste
                direction_l.append(ant_data_l[2])
                # Ajouter les directions à la liste
                last_positions.append(last_positions_l)
                # Ajouter les dernières positions à la liste

                food_counter += food_counter_l
                # Ajouter le compteur de nourriture

                if food_counter == 1 and not snapshop_taken:
                    # Si le compteur de nourriture est égal à 1 et que la capture d'écran n'a pas été prise
                    pg.image.save(screen, "MyFirstFood3_1.png")
                    # Sauvegarder l'image
                    snapshop_taken = True
                    # Changer le booléen de la capture d'écran

            age_l = np.concatenate(age_l)
            # Concaténer la liste des âges
            loaded_l = np.concatenate(loaded_l)
            # Concaténer la liste des chargements
            direction_l = np.concatenate(direction_l)
            # Concaténer la liste des directions
            last_positions = np.concatenate(last_positions, axis=0)
            # Concaténer la liste des dernières positions

            ants.age = age_l
            # Mettre à jour les âges des fourmis
            ants.is_loaded = loaded_l
            # Mettre à jour les chargements des fourmis
            ants.directions = direction_l
            # Mettre à jour les directions des fourmis

            for i, last_position in enumerate(last_positions):
                # Pour chaque dernière position
                ants.historic_path[i, ants.age[i], :] = last_position
                # Mettre à jour les dernières positions des fourmis

            pherom.display(screen)
            # Afficher les phéromones
            screen.blit(mazeImg, (0, 0))
            # Afficher l'image du labyrinthe
            ants.display(screen)
            # Afficher les fourmis
            pg.display.update()
            # Mettre à jour l'affichage
            end = time.time()
            # Initialiser la fin du temps
           
            print(f"FPS : {1. / (end - deb):6.2f}, nourriture : {food_counter:7d}", flush=True)
            # Afficher le nombre de FPS et le compteur de nourriture
            deb = time.time()
            # Initialiser le début du temps
