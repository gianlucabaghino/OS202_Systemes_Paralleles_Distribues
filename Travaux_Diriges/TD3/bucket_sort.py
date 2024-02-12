# OS202 TP3
# Gianluca Baghino Gomez

# Importer la bibliothèque MPI 
# pour la communication entre les processus
from mpi4py import MPI

# Importer la bibliothèque numpy 
# pour la manipulation de tableaux
import numpy as np

# Initialiser l'environnement MPI. 
# Cela crée un groupe de processus 
# qui peuvent communiquer entre eux.
comm = MPI.COMM_WORLD

# Obtenir le nombre total de processus dans le groupe. 
# Cela est utile pour diviser le travail entre les processus.
size = comm.Get_size()

# Obtenir le rang du processus actuel dans le groupe. 
# Le rang est un identifiant unique pour chaque processus.
rank = comm.Get_rank()

# Initialiser le générateur de nombres 
# aléatoires avec le rang du processus. 
# Cela garantit que chaque processus génère 
# un ensemble différent de nombres aléatoires.
np.random.seed(rank)

# Générer un tableau de 10 nombres aléatoires entre 0 et 100.
data = np.random.randint(0, 100, size=10)

# Trier le tableau localement en utilisant la méthode 'sort()' de numpy. 
# Cela trie le tableau en place, c'est-à-dire qu'il modifie le tableau 
# original au lieu de créer un nouveau tableau trié.
data.sort()

# Si le rang du processus est 0 (c'est-à-dire s'il s'agit du processus racine), 
# créer un tableau vide pour recevoir les données de tous les autres processus. 
# Le tableau doit être suffisamment grand pour contenir toutes les données, 
# donc sa taille est le nombre de processus multiplié par la taille du tableau de chaque processus.
if rank == 0:
    recv_data = np.empty(size*10, dtype=int)
# Si le rang du processus n'est pas 0, définir 'recv_data' comme 'None' 
# car ces processus n'ont pas besoin de recevoir des données.
else:
    recv_data = None

# Chaque processus envoie son tableau trié au processus 0. 
# La méthode 'Gather' de MPI rassemble les tableaux de tous les processus 
# et les stocke dans 'recv_data' sur le processus 0.
comm.Gather(sendbuf=data, recvbuf=recv_data, root=0)

# Si le rang du processus est 0, trier le tableau 'recv_data' qui contient maintenant 
# tous les nombres de tous les processus, et imprimer le tableau trié.
if rank == 0:
    recv_data.sort()
    print("Tableau trié : ", recv_data)
    
'''
$ mpiexec -n 8 python3 bucket_sort.py

On utilise l'interface de passage de messages (MPI) pour effectuer un 
tri par répartition (bucket sort) en parallèle sur plusieurs processus. 

1. Initialisation de MPI : 
Lorsque on exécute le programme avec 'mpiexec -n 8', 
MPI crée 8 processus qui peuvent communiquer entre eux. 
Chaque processus a un rang unique, qui est un nombre de 0 à 7 dans ce cas.

2. Génération de données : 
Chaque processus génère son propre tableau de 10 nombres aléatoires entre 0 et 100. 
Le générateur de nombres aléatoires est initialisé avec le rang du processus 
pour garantir que chaque processus génère un ensemble différent de nombres.

3. Tri local : 
Chaque processus trie ensuite son tableau localement. 
Cela est fait en utilisant la méthode 'sort()' de numpy, 
qui implémente un algorithme de tri efficace.

4. Rassemblement des données : 
Si le rang du processus est 0 (c'est-à-dire s'il s'agit du processus racine), 
il crée un tableau vide pour recevoir les données de tous les autres processus. 
Sinon, il définit 'recv_data' comme 'None'. Ensuite, chaque processus envoie 
son tableau trié au processus 0 en utilisant la fonction 'comm.Gather'. 
Cette fonction rassemble les tableaux de tous les processus 
et les stocke dans 'recv_data' sur le processus 0.

5. Tri final : 
Si le rang du processus est 0, il trie le tableau 'recv_data' 
qui contient maintenant tous les nombres de tous les processus, 
et l'imprime, ce qui donne le tableau trié final.

Résultat :

Tableau trié :  
[ 0  1  1  1  3  5  7  8  9  9  9 10 10 12 14 15 16 16 19 21 21 22 23 23
 24 25 25 27 30 34 36 37 40 41 43 44 46 47 47 49 50 55 56 57 58 61 62 62
 64 64 67 67 67 68 69 72 72 72 72 73 73 74 75 75 75 78 79 79 80 80 82 83
 83 84 87 87 92 94 99 99]

Le tableau contient 80 nombres, ce qui est correct car on a lancé le programme avec 8 processus 
(comme indiqué par mpiexec -n 8), et chaque processus a généré un tableau de 10 nombres aléatoires.

Les nombres dans le tableau sont triés dans l'ordre croissant, 
ce qui indique que l'algorithme de tri a fonctionné correctement. 
Il est également intéressant de noter que certains nombres 
apparaissent plus d'une fois dans le tableau. C'est normal 
car chaque processus génère ses nombres aléatoirement et 
indépendamment des autres processus, donc il peut y avoir des doublons.
'''