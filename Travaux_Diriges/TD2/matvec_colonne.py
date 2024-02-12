import numpy as np
import multiprocessing

# Dimension du problème
dim = 120
# Nombre de tâches parallèles
nbp = 4

# Initialisation de la matrice A
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])

# Fonction pour calculer le 
# produit matrice-vecteur par colonne
def compute_column_product(start_col, end_col):
    # Initialisation du vecteur résultat partiel
    partial_result = np.zeros(dim)
    # Calcul du produit matrice-vecteur 
    # pour chaque colonne assignée
    for col in range(start_col, end_col):
        partial_result += A[:, col] * u[col]
    return partial_result

if __name__ == '__main__':
    # Calcul du nombre de colonnes par tâche
    cols_per_task = dim // nbp
    # Création des arguments pour chaque tâche
    task_args = [(i * cols_per_task, (i+1) * cols_per_task)
                 for i in range(nbp)]

    # Création d'un pool de processus
    with multiprocessing.Pool(processes=nbp) as pool:
        # Calcul parallèle du 
        # produit matrice-vecteur par colonne
        results = pool.starmap(compute_column_product, task_args)

    # Assemblage des résultats partiels 
    # pour obtenir le vecteur résultant complet
    v = np.sum(results, axis=0)
    print("Vecteur résultant du produit matrice-vecteur par colonne :\n", v)

'''
Vecteur résultant du produit matrice-vecteur par colonne :
 [583220. 576080. 569060. 562160. 555380. 548720. 542180. 535760. 529460.
 523280. 517220. 511280. 505460. 499760. 494180. 488720. 483380. 478160.
 473060. 468080. 463220. 458480. 453860. 449360. 444980. 440720. 436580.
 432560. 428660. 424880. 421220. 417680. 414260. 410960. 407780. 404720.
 401780. 398960. 396260. 393680. 391220. 388880. 386660. 384560. 382580.
 380720. 378980. 377360. 375860. 374480. 373220. 372080. 371060. 370160.
 369380. 368720. 368180. 367760. 367460. 367280. 367220. 367280. 367460.
 367760. 368180. 368720. 369380. 370160. 371060. 372080. 373220. 374480.
 375860. 377360. 378980. 380720. 382580. 384560. 386660. 388880. 391220.
 393680. 396260. 398960. 401780. 404720. 407780. 410960. 414260. 417680.
 421220. 424880. 428660. 432560. 436580. 440720. 444980. 449360. 453860.
 458480. 463220. 468080. 473060. 478160. 483380. 488720. 494180. 499760.
 505460. 511280. 517220. 523280. 529460. 535760. 542180. 548720. 555380.
 562160. 569060. 576080.]

'''