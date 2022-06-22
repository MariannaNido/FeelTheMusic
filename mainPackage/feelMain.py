"""
Project Name: ~Feel The Music: What do you "need" to listen?~
File Name: feelMain.py
Utility: Python file rappresentante il main del programma.
Author: Nido Marianna
"""

# Librerie utilizzate
from datasetUtils import datasetHandler as dh
from surveyUtils import surveyHandler as sh
from classificationUtils import classifiersHandler as ch
import time

# Variabili utili
n_neighbors = 5
tree_depth = 5
forest_depth = 10
kf_splits = 3


# Acquisizione dati
music_dataset = dh.retrieve_data()
dh.info_dataset(music_dataset)


# Questionario
sh.info_survey()
user_answers = [sh.start_survey()]


# Predizione con KNN
knn_model = ch.k_nearest_neighbors(music_dataset, n_neighbors)
knn_pred = knn_model.predict(user_answers)

print("Il classificatore k-Nearest Neighbor, con " + str(n_neighbors) + " vicini, ha predetto per te il seguente genere musicale: " + str(knn_pred) + "\n")
print("-----------------------------------------------------------------------------------------------------------------")
time.sleep(1)


# Predizione con GNB
gnb_model = ch.gaussian_naive_bayes(music_dataset)
gnb_pred = gnb_model.predict(user_answers)

print("Il classificatore Gaussian Naive Bayes ha predetto per te il seguente genere musicale: " + str(gnb_pred) + "\n")
print("-----------------------------------------------------------------------------------------------------------------")
time.sleep(1)


# Predizione con SVM
svm_model = ch.support_vector_machine(music_dataset)
svm_pred = svm_model.predict(user_answers)

print("Il classificatore Support Machine Vector ha predetto per te il seguente genere musicale: " + str(svm_pred) + "\n")
print("-----------------------------------------------------------------------------------------------------------------")
time.sleep(1)


# Predizione con DT
dt_model = ch.decision_tree(music_dataset, max_depth=tree_depth)
dt_pred = dt_model.predict(user_answers)

print("Il classificatore Decision Tree, con profondità massima " + str(tree_depth) + ", ha predetto per te il seguente genere musicale: " + str(dt_pred) + "\n")
print("-----------------------------------------------------------------------------------------------------------------")
time.sleep(1)


# Predizione con RF
rf_model = ch.random_forest(music_dataset, max_depth=forest_depth)
rf_pred = rf_model.predict(user_answers)

print("Il classificatore Random Forest, con profondità massima " + str(forest_depth) + ", ha predetto per te il seguente genere musicale: " + str(rf_pred) + "\n")
print("-----------------------------------------------------------------------------------------------------------------")
time.sleep(1)


# Valutazioni
ch.model_eval_metrics(knn_model, "k-Nearest Neighbor", music_dataset, kf_splits)
ch.model_eval_metrics(gnb_model, "Gaussian Naive Bayes", music_dataset, kf_splits)
ch.model_eval_metrics(svm_model, "Support Machine Vector", music_dataset, kf_splits)
ch.model_eval_metrics(dt_model, "Decision Tree "+str(tree_depth), music_dataset, kf_splits)
ch.model_eval_metrics(rf_model, "Random Forest "+str(forest_depth), music_dataset, kf_splits)