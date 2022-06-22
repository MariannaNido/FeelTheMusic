"""
Project Name: ~Feel The Music: What do you "need" to listen?~
File Name: surveyHandler.py
Utility: Python file contenente funzioni e codice che permettono la somministrazione del questionario.
Author: Nido Marianna
"""

# Librerie utilizzate
import time

# Variabili utili
questions = ["A. Di solito, che brani preferisci?",
             "B. Se passasse in radio, in questo momento, un brano ritmato, cosa faresti?",
             "C. Quale di queste affermazioni ti descrive meglio in questo momento?",
             "D. In base a ciò che stai facendo, con quale di queste affermazioni ti trovi più d\'accordo?",
             "E. In questo momento, di cosa avresti più bisogno?",
             "F. Che tipo di musica preferisci ascoltare di solito?",
             "G. In questo momento, che tipo di traccia ascolteresti volentieri?",
             "H. In questo momento, come definiresti il tuo umore?"]

answers = [
    ["1. Brani molto caotici\n", "2. Brani movimentati\n", "3. Una via di mezzo tra il movimentato e il tranquillo\n",
     "4. Brani tranquilli\n", "5. Brani completamente acustici\n"],
    ["1. Nulla, probabilmente spegnerei la radio\n",
     "2. Nulla, continuerei a fare ciò che sto facendo senza alcuna differenza\n",
     "3. Mi piacerebbe molto, sarebbe un giusto sottofondo\n", "4. Attirerebbe parecchio la mia attenzione\n",
     "5. Mi coinvolgerebbe molto, probabilmente approfitterei per fare una pausa e godermi di più il ritmo\n"],
    ["1. Sono molto stanca, energie ai minimi storici\n", "2. Sono abbastanza provata, è stata una giornata pesante\n",
     "3. Sto bene, ho abbastanza energia per fare altre cose\n",
     "4. Sono abbastanza carica, pronta ad affrontare la giornata\n", "5. Sono super energica oggi, scattante\n"],
    ["1. Sto facendo cose che richiedono poca concentrazione, potrei canticchiare sulle note del mio brano preferito\n",
     "2. Sto facendo cose mediamente complicate, un po\' di musica però non guasta mai\n",
     "3. Qualsiasi tipo di musica non influenzerebbe ciò che sto facendo adesso\n",
     "4. Ho bisogno di restare concentrata, le parole delle canzoni mi distrarrebbero\n",
     "5. Se proprio ci deve essere della musica, che sia strumentale e di sottofondo\n"],
    ["1. Di più silenzio possibile\n", "2. Di non avere troppi rumori intorno\n",
     "3. Di nulla in particolare, quel che accade intorno a me non mi crea fastidio\n",
     "4. Di qualche traccia che mi tenga compagnia in sottofondo, qualunque essa sia\n",
     "5. Di spegnere i pensieri per un po\', e lasciarmi trasportare da brani ad alto volume\n"],
    ["1. Brani per niente conosciuti\n", "2. Brani poco conosciuti\n", "3. Qualsiasi tipo di brano va bene\n",
     "4. Tendo a scegliere brani spesso già sentiti\n", "5. Se un brano non è tra i più ascoltati, non fa per me\n"],
    ["1. Traccia contenente solo cantato\n", "2. Traccia contenente più cantato che parlato\n",
     "3. Il bilanciamento tra parlato e cantato sarebbe indifferente (es: musica rap)\n",
     "4. Traccia contenente più parlato che cantato\n", "5. Traccia contenente solo parlato (es: talk show)\n"],
    ["1. Non sono proprio dell\'umore giusto\n", "2. Non sono al settimo cielo, però neanche troppo giù di morale\n",
     "3. È una giornata come le altre, nulla di particolare\n", "4. Mi sento bene, sono nel mood giusto\n",
     "5. Mi sento invaso/a da energia positiva, mi sento solare\n"]]


def info_survey():
    """
    Funzione che stampa le indicazioni generali per il questionario.
    """

    print("### Sta per iniziare il questionario... ###")
    print("### Per rispondere, basterà inserire un numero da 1 a 5, corrispondente alla risposta che si vuole fornire. ###")
    print("### Ti ricordiamo che il programma cercherà di associare il genere musicale più fedele alle risposte date. ###")
    print("-----------------------------------------------------------------------------------------------------------------")

    time.sleep(3)


def start_survey():
    """
    Funzione che somministra il questionario all'utente, salvandone le risposte.
    Il questionario è composto di 8 domande, una per ogni feature di input ricavata dal dataset,
    e per ogni domanda è possibile dare una tra 5 risposte, tante quante sono le categorie utilizzate
    per trasformare le feature.
    ---------------------------------------------------------------------------------------------------------
        :return user_answers -> Risposte dell'utente al questionario
    """

    user_answers = list()

    print("INIZIA IL QUESTIONARIO\n")

    for each_question, each_answer in zip(questions, answers):
        answer = input(each_question + '\n\t' + '\t'.join(each_answer) + '\n> ')
        user_answers.append(int(answer))
        print("---------------------------------------------------")

    print("FINE DEL QUESTIONARIO")
    print("-----------------------------------------------------------------------------------------------------------------")

    return user_answers
