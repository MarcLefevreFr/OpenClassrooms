import os
import numpy as np
import tensorflow as tf
import pickle
import tensorflow
import pandas as pd
import datetime as dt

from PIL import Image
from imgaug import augmenters as iaa

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as prepro_in
from tensorflow.keras.applications.efficientnet import preprocess_input as prepro_eff

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# En l'état, ce script à exécuter en racine :
# - d'un répertoire "assets" contenant les modèles et outils utilisés.
# - d'un répertoire "data" contenant les photos de chiens à identifier.
# - les autres répertoires seront créés par le script s'ils n'existent pas

# Ces réglages peuvent être changés en modifiant les constantes qui suivent :

ASSETS = "assets/"
DATA = "data"
PREDICTIONS = "predictions/"
TEMP = "temp"

# Nom du fichier *.csv de sauvegarde des prédictions
NAME = "batch_01"

# Création des répertoires

for rep in [PREDICTIONS, TEMP, TEMP + "/temp"]:

	if os.path.exists(rep) == False  :

		os.makedirs(rep)

# Choix du modèle parmis les deux modèles ré-entrainés disponibles

modeles_dispo = ["INCEPTION", "EFFNET30"]

# Compléter le choix en indiquant une des deux valeurs en maj ci-dessus dans la ligne ci-dessous :

# *****DANS CETTE DEMO, POUR DES RAISONS PRATIQUES, SEUL EFFNET30 EST DISPONIBLE*****

choix = "EFFNET30" # indiquer INCEPTION ou EFFNET30 entre les guillemets

if choix == "INCEPTION" :

	MOD_PATH = ASSETS + "mod_inception"
	SIZE = (299,299)
	PREP = prepro_in

else :

	MOD_PATH = ASSETS + "mod_eff30_ftb.h5"
	SIZE = (224,224)
	PREP = prepro_eff


# Preprocessing initial des images.
# les images originales du répertoire "data" sont :
# - passées en mode RGB (au cas où...)
# - réduite en forme carrée-centrée selon leur plus petite longueur
# - sauvegardées sous cette forme dans le répertoire "temp"
# - En même temps, on crée une liste des noms des photos

crop = iaa.size.CropToSquare(position = "center")
liste_photos = []

for file in os.listdir(DATA):

	liste_photos.append(file)

	with Image.open(DATA + "/" + file) as image :

		if image.mode == "RGBA" :

			image = image.convert("RGB")

		image = np.asarray(image)
		image = crop.augment_image(image)
		image = Image.fromarray(image)
		image.save(TEMP + "/temp/" + file, format = "jpeg")

# Création du générateur d'images utilisant la fonction de preprocessing adaptée au modèle utilisé
gen = ImageDataGenerator(preprocessing_function = PREP)

# "aspirateur" d'image qui va chercher les images dans le répertoire TEMP afin d'effectuer les prédictions
aspirateur = gen.flow_from_directory(directory = TEMP, shuffle = False, target_size = SIZE, class_mode = None)

# Chargement du modèle, choisir INCEPTION ou EFFNET30
model = load_model(MOD_PATH)

# Récupération des prédictions brutes
preds = model.predict(aspirateur)

# On en déduit la liste des races prédites
labels = pickle.load(open(ASSETS + "dict_labels.pickle", "rb"))
labels = list(labels.keys())

races_predites = []

for p in preds :

	race = labels[np.argmax(p)]
	races_predites.append(race)

# Création d'un dataframe liste regroupant noms des photos et la race prédite correspondante.
predictions = pd.DataFrame(list(zip(liste_photos, races_predites)), columns = ['Photos', "Races"])

now = dt.datetime.now()
f_now = now.strftime("_on_%d_%m_%Y_-_%H_%M")

predictions.to_csv(PREDICTIONS + NAME + f_now + ".csv")







































