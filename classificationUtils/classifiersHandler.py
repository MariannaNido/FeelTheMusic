"""
Project Name: ~Feel The Music: What do you "need" to listen?~
File Name: classifiersHandler.py
Utility: Python file contenente funzioni utili alla classificazione.
         In particolare, si è deciso di utilizzare i seguenti classificatori:
            - K-Nearest Neighbor
            - Gaussian Naive Bayes
            - Support Vector Machine, nello specifico SVC (Support Vector Classification)
            - Decision Tree (Classification Tree)
Author: Nido Marianna
"""

# Librerie utilizzate
import time
from datasetUtils import datasetHandler
from figureUtils import figureHandler
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def k_nearest_neighbors(dataset, n_neighbors):
    """
    Funzione che applica sul dataset l'algoritmo K-Nearest Neighbors, e ne ritorna il modello addestrato,
    pronto per effettuare predizioni.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset
        :param n_neighbors -> Numero di vicini per l'algoritmo KNN
        :return model -> Il modello addestrato sul dataset
    """

    x, y = datasetHandler.split_dataset(dataset)
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    model = knn.fit(x, y)

    return model


def gaussian_naive_bayes(dataset):
    """
    Funzione che applica sul dataset l'algoritmo Gaussian Naive Bayes, e ne ritorna il modello addestrato,
    pronto per effettuare predizioni.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset
        :return model -> Il modello addestrato sul dataset
    """

    x, y = datasetHandler.split_dataset(dataset)
    gnb = GaussianNB()
    model = gnb.fit(x, y)

    return model


def support_vector_machine(dataset):
    """
    Funzione che applica sul dataset l'algoritmo Support Vector Machine, in particolare per la classificazione,
    e ne ritorna il modello addestrato, pronto per effettuare predizioni.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset
        :return model -> Il modello addestrato sul dataset
    """

    x, y = datasetHandler.split_dataset(dataset)
    svc = svm.SVC(kernel='linear')
    model = svc.fit(x, y)

    return model


def decision_tree(dataset, max_depth):
    """
    Funzione che applica sul dataset l'algoritmo Decision Tree, e ne ritorna il modello addestrato,
    pronto per effettuare predizioni. In questo caso, si parla di Classification Tree.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset
        :param max_depth -> Profondità massima del Decision Tree
        :return model -> Il modello addestrato sul dataset
    """

    x, y = datasetHandler.split_dataset(dataset)
    tree = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
    model = tree.fit(x, y)

    return model


def random_forest(dataset, max_depth):
    """
    Funzione che applica sul dataset l'algoritmo Random Forest, e ne ritorna il modello addestrato,
    pronto per effettuare predizioni. In questo caso, si parla di Random Forest Classifier.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset
        :param max_depth -> Profondità massima del Random Forest
        :return model -> Il modello addestrato sul dataset
    """

    x, y = datasetHandler.split_dataset(dataset)
    rtree = RandomForestClassifier(random_state=0, max_depth=max_depth)
    model = rtree.fit(x, y)

    return model


def model_eval_metrics(model, label, dataset, cv_splits):
    """
    Funzione che stampa le metriche per la valutazione del modello in input. Nello specifico, stampa il
    classification report e la confusion matrix per documentazione e, singolarmente, accuracy, precision,
    recall, f1-score.
    La funzione fa uso della Stratified k-Fold Cross Validation per valutare il modello.
    ---------------------------------------------------------------------------------------------------------
        :param model -> Modello di cui si vogliono conoscere le metriche
        :param label -> Etichetta per il modello che si sta valutando
        :param dataset -> Il dataset su cui lavorare
        :param cv_splits -> Numero di splits per la k-Fold Cross Validation
    """

    kf = StratifiedKFold(n_splits=cv_splits, random_state=1, shuffle=True)

    x, y = datasetHandler.split_dataset(dataset)

    prediction = cross_val_predict(estimator=model, X=x, y=y, cv=kf)

    print("### Metriche per classificatore " + str(label) + " ###\n")

    # Stampa classification report e confusion matrix
    print("## " + str(label) + " Classification report ##\n")
    print(classification_report(y_true=y, y_pred=prediction, zero_division=0))
    print("-----------------------------------------------------------------------------------------------------------------")
    time.sleep(2)

    print("...Generazione della Confusion Matrix...")
    time.sleep(1)
    figureHandler.show_confusion_matrix(model, y, prediction, label)
    print("-----------------------------------------------------------------------------------------------------------------")
    time.sleep(2)

    # Stampa metriche singolarmente
    print("## Singole metriche ##")
    print("\t#Accuracy -> ", accuracy_score(y_true=y, y_pred=prediction))
    print("\t#Precision -> ", precision_score(y_true=y, y_pred=prediction, average='macro', zero_division=0))
    print("\t#Recall -> ", recall_score(y_true=y, y_pred=prediction, average='macro', zero_division=0))
    print("\t#F1-Score -> ", f1_score(y_true=y, y_pred=prediction, average='macro', zero_division=0))
    print("-----------------------------------------------------------------------------------------------------------------")
    time.sleep(2)
