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
tree_depth = None

# Acquisizione dati
music_dataset = dh.retrieve_data()
# dh.info_dataset(music_dataset)

# Questionario
sh.info_survey()
user_answers = [sh.start_survey()]

# Predizione con KNN + metriche
knn_model = ch.k_nearest_neighbors(music_dataset, n_neighbors)
#knn_pred = knn_model.predict(user_answers)

#print("Il classificatore K-Nearest Neighbor, con " + str(n_neighbors) + " vicini, ha predetto per te il seguente genere musicale: " + str(knn_pred))
# print("\nDi seguito, le metriche per valutare il classificatore:\n")
# ch.model_metrics(knn_model, "KNN", music_dataset)
print("knn k-fold cross validation\n")
ch.kfold_cross_valid(music_dataset, 5, knn_model)
time.sleep(2)

"""
# Predizione con GNB + metriche
gnb_model = ch.gaussian_naive_bayes(music_dataset)
gnb_pred = gnb_model.predict(user_answers)

print("Il classificatore Gaussian Naive Bayes ha predetto per te il seguente genere musicale: " + str(gnb_pred))
# print("\nDi seguito, le metriche per valutare il classificatore:\n")
# ch.model_metrics(gnb_model, "GNB", music_dataset)
time.sleep(2)
"""

# Predizione con SVM + metriche
svm_model = ch.support_vector_machine(music_dataset)
svm_pred = svm_model.predict(user_answers)
print("svm k-fold cross validation\n")
ch.kfold_cross_valid(music_dataset, 5, svm_model)

#print("Il classificatore Support Machine Vector ha predetto per te il seguente genere musicale: " + str(svm_pred))
# print("\nDi seguito, le metriche per valutare il classificatore:\n")
# ch.model_metrics(svm_model, "SVM", music_dataset)
time.sleep(2)
"""
# Predizione con DT + metriche
dt_model = ch.decision_tree(music_dataset, tree_depth)
dt_pred = dt_model.predict(user_answers)

print("Il classificatore Decision Tree, con profondità massima " + str(tree_depth) + ", ha predetto per te il seguente genere musicale: " + str(dt_pred))
# print("\nDi seguito, le metriche per valutare il classificatore:\n")
# ch.model_metrics(dt_model, "DT "+str(tree_depth), music_dataset)
time.sleep(2)
"""

"""
# Predizione con CNB + metriche
cnb_model = ch.categorical_naive_bayes(music_dataset)
cnb_pred = cnb_model.predict(user_answers)

print("Il classificatore Categorical Naive Bayes ha predetto per te il seguente genere musicale: " + str(cnb_pred))
print("\nDi seguito, le metriche per valutare il classificatore:\n")
ch.model_metrics(cnb_model, "GNB", music_dataset)
"""
