import numpy as np


COLOR = dict(red = 1, blue = 2, green = 3, yellow = 4, black = 5, white = 6, pink = 7, purple = 8)

class Items:
    def __init__(self, price, is_streetwear, is_pants, is_accessory, color):
        self.price = price
        self.is_streetwear = is_streetwear
        self.is_pants = is_pants
        self.is_accessory = is_accessory
        self.color = COLOR[color.lower()]

CLOTHS = {
    "jogging_nike": Items(price=65.0, is_streetwear=1, is_pants=1, is_accessory=0, color="black"),
    "pantalon_costume": Items(price=120.0, is_streetwear=0, is_pants=1, is_accessory=0, color="blue"),
    "bonnet_carhartt": Items(price=25.0, is_streetwear=1, is_pants=0, is_accessory=1, color="yellow"),
    "t_shirt_basique": Items(price=15.0, is_streetwear=0, is_pants=0, is_accessory=0, color="white"),
    "montre_classique": Items(price=250.0, is_streetwear=0, is_pants=0, is_accessory=1, color="black")
}

class Matrice_suggestion:
    def __init__(self, cloth):
        self.items = [CLOTHS[i] for i in CLOTHS]
        self.id_cible = self.items.index(CLOTHS[cloth.lower()])
        self.matrix = self.mat()
        self.matrix_rec = self.m_suggestion()
        self.recommendation = self.suggestion()


    
    def mat(self):

        # 1. On utilise une simple liste Python classique
        M = []
        for i in self.items:
            items_attributes = [i.price, i.is_streetwear, i.is_pants, i.is_accessory, i.color]
            M.append(items_attributes)
        M = np.array(M)
        #linearisation
        normes = np.linalg.norm(M, axis=1, keepdims=True)

        # Petite astuce de pro : on ajoute une valeur minuscule (1e-9) aux normes 
        # pour éviter de diviser par zéro si un vêtement n'a aucun attribut
        normes = np.maximum(normes, 1e-9)

        # 2. On normalise la matrice en divisant chaque valeur par la norme de sa ligne
        M_normalisee = M / normes
        return M_normalisee
        
    def m_suggestion(self):
        return self.matrix @ self.matrix.T
    
    def suggestion(self):
        id_cible = self.id_cible

        # On isole la ligne correspondante dans ta matrice de similarités
        scores = self.matrix_rec[id_cible]
        # On veut les 4 meilleurs (le vêtement cible + les 3 recommandations)
        K = 4

        # argpartition place les K plus petits à gauche. 
        # Pour avoir les plus grands, on cherche les -K éléments.
        indices_top_K_desordonnes = np.argpartition(scores, -K)[-K:]

        # À ce stade, on a les 4 meilleurs index, mais ils ne sont pas dans l'ordre entre eux.
        # On récupère leurs scores pour les trier correctement :
        scores_top_K = scores[indices_top_K_desordonnes]

        # On trie ce tout petit groupe de 4 éléments du plus grand au plus petit
        tri_final = np.argsort(scores_top_K)[::-1]
        indices_finaux = indices_top_K_desordonnes[tri_final]

        # On retire le vêtement cible (le premier)
        top_3_indices_rapides = indices_finaux[1:]

        rec = []

        for i in top_3_indices_rapides:
            for k in CLOTHS:
                if CLOTHS[k] == i:
                    rec.append(k)
        return rec

    
