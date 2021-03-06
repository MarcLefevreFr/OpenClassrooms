
import pandas as pd
import numpy as np
import datetime as dt
import os
import re
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer


# Définition d'une classe **Pipeline** servant à automatiser le formatage de nouvelles données devant 
# s'ajouter aux données originales.


PATH = "Data/"
MODEL_PATH = "Data/model_rfc.pickle"

LISTE_COL = ['code', 'url', 'countries_en','product_name', 'brands', 'allergens', 'nova_group', 
             'additives_en', 'additives_n', 'nutriscore_score', 'nutriscore_grade', 
             'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n', 'energy_100g', 
             'saturated-fat_100g', 'trans-fat_100g', 'fat_100g', 'cholesterol_100g', 
             'carbohydrates_100g', 'TSu', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 
             'vitamin-a_100g', 'vitamin-c_100g', 'calcium_100g', 'iron_100g']

LISTE_FEATS = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'TSu', 'proteins_100g', 'salt_100g', 'sugars_100g']

LISTE_FR = ["France", "French", "french", "france", "français", "Français", "française", "Française", 
            "FR", "fr", "Fr", "Belgique", "belgique", "belgium", "Belgium", "belge", "Suisse", 
            "suisse", "Frankreich", "frankreich", "francais", "francaise", "Francia", "francia"]



# Classe à utiliser en entrant le nom du nouveau df d'update en toutes lettres avec l'extension.
# exemple : pipe = Pipeline("nom_df_nouvelles_données.csv")
#
# 1ere utilisation : - le df original est sauvé en "OpenFoodFacts_raw.csv"
#                    - le df modifié est sauvé sous "df.csv"
# Xème utilisation : - idem, mais en plus :
#                    - Le df original précédent est archivé en 'OFF_raw_{*DATE*}.csv'
#                    - Le df modifié original est archivé en "df_remplace_le_{*DATE*}.csv"



class Pipeline():
    
    def __init__(self, df):
        
        self.df = self.load_df(df)
        self.date = dt.datetime.now().strftime("%d-%m-%Y_%H_%M-%S")
    	
    
    
    # Chargement des DF à partir de leurs noms
    
    def load_df(self, df):
        
        # chargement que si la DF existe
        if df :
            file = os.path.join(PATH, df) 
            return pd.read_csv(file, sep = "\t", encoding = "utf-8")
        
    
    
    # Ajout des nouvelles données non modifiées aux anciennes données non modifiées
    
    def pre(self):
        
        # verification d'existence d'une version antérieure de "OpenFoodFacts_raw.csv"
        if "OpenFoodFacts_raw.csv" in os.listdir(PATH):
            
            old_raw_path = os.path.join(PATH, "OpenFoodFacts_raw.csv")
            old_raw = pd.read_csv(old_raw_path)
            
            # vérification sommaire d'homogénéité des DF
            if old_raw.shape[1] != self.df.shape[1] :
                
                return "Les DF ne sont pas homogènes"
            
            # construction nouveau df_raw, sauvegardes de l'ancien avec un autre nom et du nouveau
            else :
                       
                new_raw = pd.concat([old_raw, self.df], ignore_index = True)
                old_raw.to_csv(os.path.join(PATH, f"OFF_raw_{self.date}.csv"))
                new_raw.to_csv(os.path.join(PATH, "OpenFoodFacts_raw.csv"), index = False)
        
        # si pas de "OpenFoodFacts_raw.csv" existant, sauvegarde simple des nouvelles données
        else :
            
            self.df.to_csv(os.path.join(PATH, "OpenFoodFacts_raw.csv"), index = False)
            

    
    # traitements du DF (corrections, gestion des colonnes, etc.)
    
    def traitement(self):
        
        # corrections outliers "additives_n"
        self.df = self.df[(self.df.additives_n.isnull()) | (self.df.additives_n <= 30)]

        # corrections des nutriscore_grade == "a" erronés
        df.loc[(df["nutriscore_grade"] == "a") & (df["nutriscore_score"] > -1) & (df["nutriscore_score"] <= 2), 
               "nutriscore_grade"] = "b"
        df.loc[(df["nutriscore_grade"] == "a") & (df["nutriscore_score"] > 2) & (df["nutriscore_score"] <= 10), 
               "nutriscore_grade"] = "c"
        df.loc[(df["nutriscore_grade"] == "a") & (df["nutriscore_score"] > 10) & (df["nutriscore_score"] <= 18), 
               "nutriscore_grade"] = "d"
        df.loc[(df["nutriscore_grade"] == "a") & (df["nutriscore_score"] > 18), 
               "nutriscore_grade"] = "e"

        # corrections outliers "energy_100g"
        self.df = self.df[(self.df["energy_100g"].isnull()) | (self.df["energy_100g"] <= 6200)]

        # correction "nutriments <= 100g"
        for col in self.df.columns[74:]:
            self.df = self.df[(self.df[col].isnull()) | (self.df[col] >= 0)]
            self.df = self.df[(self.df[col].isnull()) | (self.df[col] < 100)]
            
        # corrections couples de variables    
        self.df.loc[self.df["saturated-fat_100g"] > self.df["fat_100g"], "saturated-fat_100g" ] = self.df["fat_100g"]
        self.df.loc[self.df["trans-fat_100g"] > self.df["fat_100g"], "trans-fat_100g" ] = self.df["fat_100g"]
        self.df.loc[self.df["sugars_100g"] > self.df["carbohydrates_100g"], "sugars_100g" ] = self.df["carbohydrates_100g"]
            
        # création colonne "taux de sucre dand glucides"
        self.df["TSu"] = self.df["sugars_100g"] / self.df["carbohydrates_100g"]
        
        # supression de colonnes et mise en ordre des colonnes restantes
        self.df = self.df[LISTE_COL]
        
        # traitement des variables quantitatives TEXTE
        colonnes = self.df.columns[2:8].tolist()
        colonnes.remove("nova_group")
        
        for col in colonnes :
            self.df[col] = self.df[col].map(self.conv_suite_mots)
            conv_suite_mots

        # complétion du "nutriscore_grade" grâce à notre modèle
        modele = pickle.load(open(MODEL_PATH, "rb"))
                
        df_ac = self.df[self.df.nutriscore_grade.isnull()]
        
        for col in LISTE_FEATS :
            df_ac = df_ac[df_ac[col].notnull()]

        X = df_ac[feats]

        df_ac["nutriscore_grade"] = modele.predict(X)

        self.df.loc[df["nutriscore_grade"].isnull(), "nutriscore_grade"] = df_ac.nutriscore_grade

        # réduction à la sphère francophone

        self.df["countries_en"] = self.df["countries_en"].map(self.mapping_fr)
        self.df = self.df[self.df["countries_en"] == "francophone"]
        self.df = self.df.drop(["countries_en"], axis=1)

        # traitement des mots avant création du moteur

        self.df = self.df[self.df["product_name"].notnull()]
        self.df["words"] = self.df["product_name"].astype('str') + " " + self.df["brands"].fillna("").astype('str')
        
        tokenizer = nltk.RegexpTokenizer('[a-z]\w+')
        sw = stopwords.words("french")
        stemmer = FrenchStemmer()

        self.df["tokens"] = self.df["words"].map(tokenizer.tokenize)
        
        self.df["tokens"] = self.df["tokens"].map(self.sw_stem_str)

        vect = TfidfVectorizer(analyzer = "word")
		vect.fit(self.df["tokens"])
		matrix = vect.transform(self.df["tokens"])

		self.df = self.df.drop(["words", "tokens"], axis = 1)

		# Sauvegarde des éléments utilisés par le moteur (autres que le dataset)

		pickle1 = open("Data/vect.pickle", "wb")
		pickle.dump(vect, pickle1)
		pickle1.close()

		pickle2 = open("Data/matrix.pickle", "wb")
		pickle.dump(matrix, pickle2)
		pickle2.close()

		pickle3 = open("Data/tokenizer.pickle", "wb")
		pickle.dump(tokenizer, pickle3)
		pickle3.close()






    def sw_stem_str(self, l):
    
    for elt in l:
        if elt in sw :
            l.remove(elt)

    li = [stemmer.stem(w) for w in l]        
    
    ligne = " ".join(li)
    return ligne


    # méthode de traitement des variables texte
    
    def conv_suite_mots(self, s):
        
        if type(s) != float :
  
            liste = s.lower().split(", ")
            new = []

            for w in liste :

                if w.startswith("en:"):
                    w = w[3:]
                elif w.startswith("fr:"):
                    w = w[3:]
                elif w.startswith("es:"):
                    w = w[3:]
                new.append(w)

            new = list(set(new))
            return " ".join(new)
        
        
    
    # concaténation et sauvegarde du nouveau DF de travail. Renommage de l'ancien s'il existe.
    
    def post(self):
        
        # si une df existe dejà, on concate les deux, renomme l'ancienne, et sauvegarde la nouvelle version en "df.csv"
        if "df.csv" in os.listdir(PATH):
            old_file = os.path.join(PATH, "df.csv")
            df_old_mod = pd.read_csv(old_file)
            new_df = pd.concat([df_old_mod, self.df], ignore_index = True)
            df_old_mod.to_csv(os.path.join(PATH, f"df_remplace_le_{self.date}.csv"))
            new_df.to_csv(os.path.join(PATH, "df.csv"), index = False)
        
        # sinon on sauve juste self.df
        else :
            self.df.to_csv(os.path.join(PATH, "df.csv"), index = False)
    
    
    def mapping_fr(self):

    	for val in re.findall(r"[\w']+", str(value)): # utilisation d'une RE   
        	if val in LISTE_FR :
            	return "France"
    
    	return value








    
    # regroupement des étapes du traitement en une méthode à invoquer...
    
    def process(self):
        
        self.pre()
        self.traitement()
        self.post()





# Insérer le nom du nouveau fichier de données "new_data.csv" dans le cast de "Pipeline"
if __name__ == '__main__':

	pipe = Pipeline("")
	pipe.process()



