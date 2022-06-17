"""
Project Name: ~Feel The Music: What do you "need" to listen?~
File Name: classifiersHandler.py
Utility: Python file contenente funzioni utili alla classificazione.
         In particolare, si è deciso di utilizzare i seguenti classificatori:
            - K-Nearest Neighbor
            - Gaussian Naive Bayes
            - Support Vector Machine, nello specifico SVC (Support Vector Classification)
            - Decision Tree
Author: Nido Marianna
"""

# Librerie utilizzate
import time
from datasetUtils import datasetHandler
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from figureUtils import figureHandler as fh


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
    pronto per effettuare predizioni.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset
        :param max_depth -> Profondità massima del Decision Tree
        :return model -> Il modello addestrato sul dataset
    """

    x, y = datasetHandler.split_dataset(dataset)
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=max_depth)
    model = tree.fit(x, y)

    return model


def model_metrics(model, label, dataset):
    """
    Funzione che stampa il Classification Report e la Matrice di Confusione per il modello passato in input.
    ---------------------------------------------------------------------------------------------------------
        :param model -> Modello di cui si vogliono conoscere le metriche
        :param label -> Etichetta per il modello che si sta valutando
        :param dataset -> Il dataset su cui lavorare
    """

    x, y = datasetHandler.split_dataset(dataset)

    prediction = model.predict(x)

    print("### " + str(label) + " Classification report ###\n")
    time.sleep(2)
    print(classification_report(y_true=y, y_pred=prediction, zero_division=0))

    print("...Generazione della Confusion Matrix...")
    time.sleep(2)
    fh.show_confusion_matrix(model, y, prediction, label)
    print("-----------------------------------------------------------------------------------------------------------------")


# Forse (accuracy 50%)
def categorical_naive_bayes(dataset):
    """
    Funzione che applica sul dataset l'algoritmo Categorical Naive Bayes, e ne ritorna il modello addestrato,
    pronto per effettuare predizioni.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset
        :return model -> Il modello addestrato sul dataset
    """

    x, y = datasetHandler.split_dataset(dataset)
    cnb = CategoricalNB()
    model = cnb.fit(x, y)

    return model
