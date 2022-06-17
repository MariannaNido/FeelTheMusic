"""
Project Name: ~Feel The Music: What do you "need" to listen?~
File Name: feelMain.py
Utility: Python file contenente il main del programma.
Author: Nido Marianna
"""

# Librerie utilizzate
from datasetUtils import datasetHandler as dh
from surveyUtils import surveyHandler as sh
from classificationUtils import classifiersHandler as ch
import time

# Variabili utili
n_neighbors = 5
tree_depth = 3

# Acquisizione dati
music_dataset = dh.retrieve_data()
#dh.info_dataset(music_dataset)

# Questionario
#sh.info_survey()
user_answers = [sh.start_survey()]

# Predizioni
knn_model = ch.k_nearest_neighbors(music_dataset, n_neighbors)
knn_pred = knn_model.predict(user_answers)

print(user_answers)
print(knn_pred)
ch.model_metrics(knn_model, music_dataset)