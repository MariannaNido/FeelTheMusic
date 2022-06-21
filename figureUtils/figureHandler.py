"""
Project Name: ~Feel The Music: What do you "need" to listen?~
File Name: figureHandler.py
Utility: Python file contenente funzioni e codice per la stampa dei grafici utili.
Author: Nido Marianna
"""

# Librerie utilizzate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


def show_confusion_matrix(model, y_values_true, y_values_pred, label):
    """
    Funzione che mostra la Matrice di Confusione per il modello dato.
    ---------------------------------------------------------------------------------------------------------
        :param model -> Il modello da valutare
        :param y_values_true -> Valori reali
        :param y_values_pred -> Valori predetti
        :param label -> Etichetta per il modello che si sta valutando
    """
    """
    cm = confusion_matrix(y_true=y_values_true, y_pred=y_values_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    disp.plot()
    plt.xticks(np.arange(len(model.classes_)), model.classes_)
    plt.show()
    """
    matrix_title = str(label) + " Confusion Matrix"

    matrix = confusion_matrix(y_true=y_values_true, y_pred=y_values_pred, labels=model.classes_)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='YlGn',
                linecolor='black',
                linewidths=1,
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                annot=True,
                fmt='d')
    plt.title(matrix_title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(len(model.classes_)), model.classes_, rotation=20)
    plt.show()


def show_histogram(classes, values):
    """
    Funzione che mostra l'istogramma relativo alle macro classi e al numero di elementi per ognuna.
    ---------------------------------------------------------------------------------------------------------
        :param classes -> Le classi da visualizzare nell'istogramma
        :param values -> I valori per ogni classe
    """

    plt.subplots(num="Istogramma Classi/Num elementi")
    plt.bar(classes, values, color='#1DB954', edgecolor='black')
    plt.xticks(np.arange(len(classes)), rotation=20)
    plt.show()


