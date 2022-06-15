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
import matplotlib.pyplot as plt
import time
from datasetUtils import datasetHandler
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


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


def model_evaluation(model, prediction, dataset):
    """
    Funzione che stampa il Classification Report e la Matrice di Confusione per il modello passato in input.
    ---------------------------------------------------------------------------------------------------------
        :param model -> Modello di cui si vogliono conoscere le metriche
        :param prediction -> La predizione effettuata dal modello
        :param dataset -> Il dataset da cui estrarre le feature di output
    """

    x, true_dataset = datasetHandler.split_dataset(dataset)

    print("### Classification report ###\n")
    time.sleep(2)
    print(classification_report(y_true=true_dataset, y_pred=prediction, zero_division=0))
    print("------------------------------------------------------------------")

    print("### Generazione della Confusion Matrix ###")
    time.sleep(2)
    cm = confusion_matrix(y_true=true_dataset, y_pred=prediction, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()
    print("------------------------------------------------------------------")

