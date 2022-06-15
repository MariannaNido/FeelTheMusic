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

# Variabili utili
n_neighbors = 5
tree_depth = 3

# Lettura del dataset e acquisizione dati
music_dataset = dh.retrieve_data()
dh.info_dataset(music_dataset)

