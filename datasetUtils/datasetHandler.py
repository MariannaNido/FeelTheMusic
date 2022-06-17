"""
Project Name: ~Feel The Music: What do you "need" to listen?~
File Name: datasetHandler.py
Utility: Python file contenente funzioni e codice che permettono la preparazione del dataset
         ai fini della classificazione. Il dataset di partenza è stato ricavato dal sito Kaggle
         al seguente indirizzo: https://www.kaggle.com/datasets/pesssinaluca/spotify-by-generes,
         e verrà modificato per renderlo più facilmente utilizzabile dal programma stesso.
Author: Nido Marianna
"""

# Librerie utilizzate
import pandas as pd
import time
from figureUtils import figureHandler as fh

# Variabili utili
original_file_loc = "C:/Users/maria/Desktop/FeelMusic/datasetUtils/data_by_genres.csv"
new_file_loc = "C:/Users/maria/Desktop/FeelMusic/datasetUtils/excerpt_data_by_genres.csv"
main_class_genres = [["classical", "Classical"],
                     ["comedy", "Comic Sketch"],
                     ["country", "Country"],
                     ["dance", "Dance"],
                     ["hip hop", "Hip Hop"],
                     ["indie", "Indie"],
                     ["jazz", "Jazz"],
                     ["lo-fi", "Lo-Fi"],
                     ["metal", "Metal"],
                     ["pop", "Pop"],
                     ["rap", "Rap/Trap"],
                     ["reggae", "Reggae/Reggaeton"],
                     ["rock", "Rock"],
                     ["techno", "Techno"]]


def load_raw_dataset():
    """
    Funzione di lettura del dataset originale, con conseguente rimozione delle colonne che non verranno prese
    in considerazione dal programma.
    ---------------------------------------------------------------------------------------------------------
        :return clear_dataset -> Dataset contenente solo le colonne (features) utili.
    """

    print("### Acquisizione del dataset ###")
    time.sleep(3)
    print("------------------------------------------------------------------")

    raw_dataset = pd.read_csv(original_file_loc)
    clear_dataset = pd.DataFrame(raw_dataset.drop(columns=['duration_ms', 'liveness', 'key', 'mode', 'tempo']))

    return clear_dataset


def adapt_dataset(dataset, classes):
    """
    Funzione che, data la lista composta da coppie [genere, categoria], estrae dal dataset in input i soli
    generi la cui feature "genres" contiene la (sotto)stringa "genere". Ricavate le righe del dataset
    corrispondenti alla ricerca, il valore in "genres" viene sostituito dalla più generica "categoria",
    in modo che tutti i generi "simili" abbiano un unico <macro_genere> di appartenenza.
    Successivamente, i valori delle restanti feature vengono ridimensionati e suddivisi complessivamente in
    5 categorie tramite la funzione categorize_feature().
    Il dataset risultante viene epurato da eventuali duplicati, e viene aggiunto il tutto al dataset finale,
    che verrà restituito in output.
    Prima di restituire il dataset, viene effettuata un'ulteriore rimozione dei duplicati, non considerando la
    feature 'genres', in modo da evitare problemi di classificazione successivi.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset su cui si vuole operare l'adattamento
        :param classes -> Lista contenente i generi da estrarre e l'etichetta del relativo <macro_genere>
        :return new_dataset -> Nuovo dataset riformattato
    """

    new_dataset = pd.DataFrame()

    for gen, cat in classes:
        temp = pd.DataFrame()

        result = pd.DataFrame(dataset[dataset['genres'].str.contains(gen)])
        temp = pd.concat([temp, result], ignore_index=True)

        # Sostituzione del genere
        temp.loc[temp['genres'].str.contains(gen), 'genres'] = cat

        for col in list(temp.columns)[1:]:
            categorize_feature(temp, col)

        temp = temp.drop_duplicates(ignore_index=True)

        new_dataset = pd.concat([new_dataset, temp], ignore_index=True)

    new_dataset = new_dataset.drop_duplicates(subset=new_dataset.columns.difference(['genres']), ignore_index=True)

    return new_dataset


def categorize_feature(dataset, feature):
    """
    Funzione che permette di modificare i valori all'interno di un dataset, assegnando delle categorie
    ad ogni elemento per ogni feature che lo caratterizza, in modo da categorizzare i valori ed effettuare
    una sorta di standardizzazione. La categorizzazione dei valori è fatta in questo modo:
        Very Low  -> 1 -> [0.0, 0.2) / [ 0, 20) per 'popularity' / [-40, -32) per 'loudness'
        Low       -> 2 -> [0.2, 0.4) / [20, 40) per 'popularity' / [-32, -24) per 'loudness'
        Medium    -> 3 -> [0.4, 0.6) / [40, 60) per 'popularity' / [-24, -16) per 'loudness'
        High      -> 4 -> [0.6, 0.8) / [60, 80) per 'popularity' / [-16, -8) per 'loudness'
        Very High -> 5 -> [0.8, 1.0] / [80, 100] per 'popularity' / [-8, 0] per 'loudness'
    ---------------------------------------------------------------------------------------------------------
    :param dataset -> Dataset che si vuole modificare
    :param feature -> La feature da considerare per la modifica
    """

    if str(feature) == 'popularity':
        dataset.loc[(dataset[feature] >= 0) & (dataset[feature] < 20), feature] = 1
        dataset.loc[(dataset[feature] >= 20) & (dataset[feature] < 40), feature] = 2
        dataset.loc[(dataset[feature] >= 40) & (dataset[feature] < 60), feature] = 3
        dataset.loc[(dataset[feature] >= 60) & (dataset[feature] < 80), feature] = 4
        dataset.loc[(dataset[feature] >= 80) & (dataset[feature] <= 100), feature] = 5

    elif str(feature) == 'loudness':
        dataset.loc[(dataset[feature] >= -8) & (dataset[feature] <= 0), feature] = 5
        dataset.loc[(dataset[feature] >= -16) & (dataset[feature] < -8), feature] = 4
        dataset.loc[(dataset[feature] >= -24) & (dataset[feature] < -16), feature] = 3
        dataset.loc[(dataset[feature] >= -32) & (dataset[feature] < -24), feature] = 2
        dataset.loc[(dataset[feature] >= -40) & (dataset[feature] < -32), feature] = 1

    else:
        dataset.loc[(dataset[feature] >= 0.8) & (dataset[feature] <= 1), feature] = 5
        dataset.loc[(dataset[feature] >= 0.6) & (dataset[feature] < 0.8), feature] = 4
        dataset.loc[(dataset[feature] >= 0.4) & (dataset[feature] < 0.6), feature] = 3
        dataset.loc[(dataset[feature] >= 0.2) & (dataset[feature] < 0.4), feature] = 2
        dataset.loc[(dataset[feature] >= 0.0) & (dataset[feature] < 0.2), feature] = 1


def save_dataset(dataset, save_path):
    """
    Funzione che salva il nuovo dataset creato come file CSV nel path specificato.
    ---------------------------------------------------------------------------------------------------------
    :param dataset -> Dataset che si intende salvare
    :param save_path -> Path in cui si intende salvare il dataset
    """

    dataset.to_csv(save_path, index=False)


def retrieve_data():
    """
    Funzione "main" che permetterà di recuperare il dataset pronto per la classificazione, e lo salva in
    automatico al path specificato in base alla funzione save_dataset().
    ---------------------------------------------------------------------------------------------------------
    :return dataset -> Dataset pronto per essere utilizzato
    """

    raw = load_raw_dataset()
    dataset = adapt_dataset(raw, main_class_genres)
    save_dataset(dataset, new_file_loc)

    return dataset


def info_dataset(dataset):
    """
    Funzione che stampa alcune informazioni sul dataset che si utilizza
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset di cui si vogliono conoscere le informazioni
    """

    print("\t### Ecco alcune informazioni riguardanti il dataset acquisito ###\n")
    time.sleep(1)

    print("Elementi totali contenuti nel dataset: " + str(len(dataset)) + ".")
    print("------------------------------------------------------------------")
    time.sleep(1)

    print("Le features del dataset:\n\t#" + "\n\t#".join(list(dataset.columns)))
    print("------------------------------------------------------------------")
    time.sleep(1)

    print("Numero di elementi per ogni macro classe:")
    for c, v in zip(get_macro_classes_names(main_class_genres), get_count_for_classes(dataset)):
        print("\t" + str(c) + " -> " + str(v))
    time.sleep(1)

    print("\n...Apertura istogramma con elementi relativi ad ogni macro classe individuata...")
    fh.show_histogram(get_macro_classes_names(main_class_genres), get_count_for_classes(dataset))
    print("------------------------------------------------------------------")
    time.sleep(1)

    print("Categorie in cui sono stati riformattati i valori:\n"
          "\tVery Low  -> 1\n\tLow       -> 2\n\tMedium    -> 3\n\tHigh      -> 4\n\tVery High -> 5")
    print("------------------------------------------------------------------")
    time.sleep(2)


def split_dataset(dataset):
    """
    Funzione che divide il dataset in feature di input e feature di output.
    ---------------------------------------------------------------------------------------------------------
    :param dataset -> Dataset da dividere
    :return input_features -> Features di input, in questo caso le caratteristiche dei generi
    :return output_features -> Features di output, in questo caso i generi da predire
    """

    input_features = dataset.iloc[:, 1:].values
    output_features = dataset.iloc[:, 0].values

    return input_features, output_features


def get_count_for_classes(dataset):
    """
    Funzione che conta gli elementi per ogni macro classe nel dataset.
    ---------------------------------------------------------------------------------------------------------
        :param dataset -> Dataset da cui ricavare i valori
        :return values -> Valori per ogni macro classe
    """

    values = []
    grouped = dataset.groupby('genres').count()['acousticness']
    # Possiamo usare una feature qualunque al posto di 'acousticness', dato che il conteggio resta uguale

    for i in range(len(grouped)):
        values.append(grouped[i])

    return values


def get_macro_classes_names(macro_classes):
    """
    Funzione che recupera le "etichette" delle macro classi individuate.
    ---------------------------------------------------------------------------------------------------------
        :param macro_classes -> Macro classi con generi
        :return classes -> Etichette delle macro classi
    """

    classes = []

    for i in range(len(macro_classes)):
        classes.append(macro_classes[i][1])

    return classes
