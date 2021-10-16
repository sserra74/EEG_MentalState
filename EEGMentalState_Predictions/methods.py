import sys
import time
from pprint import pprint

import pydot
import pandas as pd
import numpy as np
import matplotlib
from scipy.integrate import simps
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import csv

# Dal dataFrame, prendiamo i valori del canale "col", li inseriamo in listchannel e plottiamo i valori
def addChannels(list_channels, col, list_data,idx,list_figure, list_subplt):
    #print(col)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)

    list_figure.append(fig)
    list_subplt.append((ax1, ax2, ax3))

    # Lista dei valori dei canali
    list_channels.append(list_data[:,col].tolist())

    # Rappresento graficamente i valori nella mia lista, ovvero le frequenze associate ai campioni
    ax1.plot(list_channels[idx])
    ax1.title.set_text('Signal ' + str(col))
    ax1.set_xlabel('nSamples')
    ax1.set_ylabel('Frequency [Hz]')

    return list_channels, list_figure, list_subplt


def computeSpecgram(idx, list_channels,sampleFreq, list_subplt):

    ax1,ax2,ax3 = list_subplt[idx]
    # Trasformo i valori in x da secondi a minuti
    #formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M', time.gmtime(ms)))
    #ax2.xaxis.set_major_formatter(formatter)
    #ax3.xaxis.set_major_formatter(formatter)

    # Se viene calcolato lo spettrogramma del segnale specificando solamente la frequenza di campionamento, si ottiene uno
    # spettrogramma abbastanza simile a quello mostrato in fig 6 nel paper. L'intuizione è che gli autori del paper abbiano
    # mostrato dunque lo spettrogramma del segnale senza specificare alcun tipo di parametro
    P, f, b, imm = ax2.specgram(list_channels[idx], Fs=sampleFreq, cmap='jet')
    #fig.colorbar(imm, ax=ax2).set_label('Intensity [dB]')
    #ax2.xaxis.set_major_formatter(formatter)
    ax2.set(xlabel='Time [min]', ylabel='Frequency [Hz]')

    # Primo metodo con cui genero lo spettrogramma, uso la finestra di Blackman con dimensione pari a 1920, ma di questa
    # finestra considero solamente 1024 punti tramite pad_to, come specificato nel paper. Il parametro noverlap
    # è stato inserito per avere degli step di 1 secondo tra un valore e l'altro

    Pxx, freqs, xbins, im = ax3.specgram(list_channels[idx], Fs=sampleFreq, window=np.blackman(32), NFFT=32, noverlap=31,
                                             cmap='jet')
    ax3.set_title('Spectogram')
    ax3.set(xlabel='Time [min]', ylabel='Frequency [Hz]')

    # Aggiunta colorbar per indicare l'intensità
    #fig.colorbar(im, ax=ax3).set_label('Intensity [dB]')

    # restituisco i valori
    return P, f, b, imm, Pxx, freqs, xbins, im



def average_spectral_power(sampleFreq, f, spec_values, Power):

    # Risoluzione di frequenza
    freq_res = sampleFreq / 32

    # Low e High sono i valori associati alla banda di frequenza che stiamo considerando, infatti dobbiamo calcolare
    # l'average spectral power delle frequenze contenute in ogni banda di frequenza 0.5Hz, che parte da 0 fino a 64Hz
    low, high = 0, 0.5

    # Lista contenente i booleani che saranno a True se quelle frequenze considerate stanno all'interno del range
    # specificato, altrimenti saranno False
    idx_delta = []

    # L'indice j scorre i valori booleani all'interno di idx_delta
    j = 0

    # L'indice i scorre ogni frequenza (viene escluso lo 0, come riportato nel paper), mentre k è l'indice che viene usato
    # all'interno del dizionario
    i = 1
    k = 1

    # Variabile che indica il numero di "salti" che si devono fare per passare da una banda alla successiva
    hop = 4

    #print("Frequenze")
    #pprint(f)
    # Scorriamo tutte le frequenze
    while (i < len(f)):
        # Se la frequenza considerata non supera il limite superiore della banda
        if (f[i] <= high):

            # Andiamo a settare a True tutte le frequenze che ricando dentro il range specificato
            idx_delta.append(np.logical_and(f > low, f <= high))
            # Dal momento che dobbiamo calcolare l'average spectral power di un intervallo di frequenze, è necessario
            # usare degli integrali, utilizziamo dunque il metodo simps (Simpsons' rule)

            """
            print("IDX")
            print(idx_delta)

            print("Power")
            pprint(Power[idx_delta[j]])
            """


            #countTrue = [x for x in idx_delta if x == True]

            countTrue = sum(map(lambda x: x == True, idx_delta[j]))

            #print(countTrue)
            #print("AAAA")


            if(countTrue>1):
                #print("c mag 1")
                avg_sp = simps(Power[idx_delta[j]], dx=freq_res)
            else:
                #print("c min")
                avg_sp = Power[idx_delta[j]][0]


            #print("Avg")
            #print(avg_sp)
            spec_values.append(avg_sp)
            #print("Spec")
            #print(spec_values)

            # Ciclo usato per popolare il dizionario associando alle frequenze che appartengono alla banda corrente
            # lo stesso average spectral power
            while (k <= (i + hop - 1)):
                #dict_freq[f[k]] = avg_sp
                k += 1

            # Incremento gli indici per considerare tutti gli elementi correttamente
            j += 1
            i += hop # l'indice i punterà al primo elemento della banda successiva
            k = i

            # Step 3:
            # Dal paper:
            # The frequency range was then restricted to 0–18 Hz so that only 36 frequencies were retained in the dataset
            # Per comodità, eseguiamo immediatamente lo step 3 non andando a considerare le frequenze oltre i 18Hz
            # In questo modo, otteniamo le 36 frequenze e i relativi valori come specificato
            if (high == 18):
                i = len(f)
        else:
            # Ogni volta che finiamo di esaminare una banda, aggiorniamo gli estremi e analizziamo quella successiva
            low += 0.5
            high += 0.5

    # Restituiamo la lista contenente i valori dell'average spectral power e il dizionario per la rappresentazione grafica
    return spec_values

# Metodo per il calcolo della running average con window = 15
def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'same')
    return smas

# Per ogni lista nel vettore di feature, scorriamo ogni elemento e lo inseriamo nel vettore finale delle feature, che
# sarà utilizzato per l'SVM
def finalFeatureVector(featureVector):

    feature_final = []
    for l in featureVector:
        for element in l:
            feature_final.append(element)

    return feature_final

def definingClass(rawData):

    # Mi creo un dizionario in cui gestisco meglio i dati per lavoraci sopra, genero quindi prima le chiavi
    keys = range(len(rawData["timestamp"][0, 0]))

    # Genero il dizionario, zip è un metodo che associa ad ogni elemento in keys la rispettiva riga in raw_data
    dict_timestamp = dict(zip(keys, rawData["timestamp"][0, 0]))

    # Ottengo la tabella con le colonne che mi interessano
    table_timestamp = pd.DataFrame.from_dict(dict_timestamp, orient='index')
    table_edit = table_timestamp[[3,4]]

    # Estraggo le singole colonne
    col_hour = table_edit[3]
    col_minutes = table_edit[4]

    # Creo colonna per la classe, momentaneamente piena di 0 per ogni riga
    table_edit['Class'] = np.zeros(len(table_edit))

    # Assegno la classe in base all'orario
    table_edit.loc[(col_hour == 19) & (col_minutes >= 33) & (col_minutes <= 43), 'Class'] = 1
    table_edit.loc[(col_hour == 19) & (col_minutes >= 44) & (col_minutes <= 54), 'Class'] = 2
    table_edit.loc[((col_hour == 19) & (col_minutes >= 55)) | (col_hour==20), 'Class'] = 3

    return table_edit

# Applico classificatore SVM
def computeSVM(X_train_df,X_test_df,Y_train,Y_test,root_square_error_list,square_error_list,r2_list,svm_model_linear,prediction_SVM):

    # Specifiamo la kernel function, in questo caso 'linear' (citata nel paper) per costruire il modello su
    # gli insiemi di training creati in precedenza
    #svm_model_linear = SVC(kernel='linear').fit(X_train_df, Y_train)
    #svm_model_linear = LinearSVC(max_iter=100000).fit(X_train_df, Y_train)
    svm_model_linear.fit(X_train_df, Y_train)

    predictions = svm_model_linear.predict(X_test_df)
    print(predictions)

    prediction_SVM.append(predictions.tolist())

    # Applico il modello sugli insiemi di test e score restituisce l'accuratezza rispetto agli insiemi dati
    # l'accuratezza della fold corrente viene inserita in una lista

    root_square_error_list.append(metrics.mean_squared_error(Y_test,predictions,squared=False))
    square_error_list.append(metrics.mean_squared_error(Y_test, predictions, squared=True))
    r2 = metrics.r2_score(Y_test,predictions)
    print(r2)

    if (r2 < 0):
        r2_adj = np.corrcoef(predictions, Y_test)[0, 1] ** 2
        print(r2_adj)
        r2_list.append(r2_adj)
    else:
        r2_list.append(r2)



    return root_square_error_list, square_error_list,r2_list, prediction_SVM

# Applico il classificatore definito dal parametro "classifier"
def computeClassifier(X_train_df,X_test_df,Y_train,Y_test,root_square_error_list,square_error_list,r2_list,classifier, prediction_list):

    # Alleno il modello con i dati di training
    classifier.fit(X_train_df.astype(np.float), Y_train.astype(np.float))

    # Uso il classificatore per le predizioni sui dati di test
    predictions = classifier.predict(X_test_df.astype(np.float))

    prediction_list.append(predictions.tolist())

    root_square_error_list.append(metrics.mean_squared_error(Y_test, predictions, squared=False))
    square_error_list.append(metrics.mean_squared_error(Y_test, predictions, squared=True))

    r2 = metrics.r2_score(Y_test, predictions)

    if (r2 < 0):
        r2_adj = np.corrcoef(predictions, Y_test)[0, 1] ** 2
        print(r2_adj)
        r2_list.append(r2_adj)
    else:
        r2_list.append(r2)

    # Restituisco la lista contenente le accurarezze per ciascuna fold
    return root_square_error_list, square_error_list,r2_list,prediction_list

# Metodo che calcola le metriche
def metricsPrevision(Y_test, predictions):

    # Calcolo accuratezza del classificatore in base alle predizioni e ai valori di classe di test
    accuracy = metrics.accuracy_score(Y_test.astype(np.float), predictions)

    # Il metodo precision_recall_fscore_support restituisce le metriche di interesse
    precision, recall, fscore, n = precision_recall_fscore_support(Y_test.astype(np.float), predictions, average='weighted')

    # Ottengo la confusion matrix
    cm = confusion_matrix(Y_test.astype(np.float), predictions)

    # Restituisco le varie metriche calcolate
    return accuracy, precision, recall, fscore, cm



# Restituisco accuratezza media della lista che viene passata come parametro
def getAverage(root_square_error_list_final, square_error_list_final, r2_list_final,b):

    # b è un booleano che specifica se è necessario calcolare la percentuale o meno
    if(b==True):
        rmse = round((np.mean(root_square_error_list_final) * 100),2)
        mse = round((np.mean(square_error_list_final) * 100), 2)
        r2 = round((np.mean(r2_list_final) * 100), 2)
    else:
        rmse = round(np.mean(root_square_error_list_final), 2)
        mse = round(np.mean(square_error_list_final), 2)
        r2 = round(np.mean(r2_list_final), 2)


    # Restituisco accuratezza, precision, recall e fscore media
    return rmse,mse, r2

# Passando come parametri le liste delle metriche, calcolo la media e stampo i valori
def printAverageValues(root_square_error_list_final, square_error_list_final, r2_list_final):

    a, p, r = getAverage(root_square_error_list_final, square_error_list_final, r2_list_final, True)


    print("Media RMSE records")
    print(str(a) + "%")
    print("")

    print("Media MSE records")
    print(str(p) + "%")
    print("")

    print("Media r2 records")
    print(str(r) + "%")
    print("")


# Date le metriche, queste vengono stampate
def printValues(a,p,r,fs):

    print("Media accuratezza records")
    print(str(a*100) + "%")
    print("")

    print("Media precision records")
    print(str(p*100) + "%")
    print("")

    print("Media recall records")
    print(str(r*100) + "%")
    print("")

    print("Media fscore records")
    print(str(fs*100) + "%")
    print("")

# Metodo che scrive le predizioni dei classificatori su un file comune
def writePrediction(prediction_SVM,prediction_RF,prediction_BN,prediction_KNN,prediction_LR):
    with open('Predictions.csv', 'w+', newline='') as file:
        writer = csv.writer(file)

        # Ogni predizioni è separata da dei trattini ---
        writer.writerows(prediction_SVM)
        writer.writerow("---")
        writer.writerows(prediction_RF)
        writer.writerow("---")
        writer.writerows(prediction_BN)
        writer.writerow("---")
        writer.writerows(prediction_KNN)
        writer.writerow("---")
        writer.writerows(prediction_LR)

        file.close()

# Metodo che legge il file delle predizioni e riempie le liste con le rispettive predizioni per ogni classificatore
def readPrediction():
    prediction_SVM,prediction_RF, prediction_BN, prediction_KNN, prediction_LR = [],[],[],[],[]
    i = 1

    with open('Predictions.csv', 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')

        # Ogni riga nel file csv viene inserita nella lista
        for row in csv_reader:
            if(i < 58):
                prediction_SVM.append(row)
            elif(i>58 and i < 116):
                prediction_RF.append(row)
            elif (i > 116 and i < 174):
                prediction_BN.append(row)
            elif (i > 174 and i < 232):
                prediction_KNN.append(row)
            elif (i > 232 and i < 290):
                prediction_LR.append(row)

            i += 1

    file.close()

    prediction_SVM = np.array(prediction_SVM).astype(np.float)
    prediction_RF = np.array(prediction_RF).astype(np.float)
    prediction_BN = np.array(prediction_BN).astype(np.float)
    prediction_KNN = np.array(prediction_KNN).astype(np.float)
    prediction_LR = np.array(prediction_LR).astype(np.float)


    return prediction_SVM,prediction_RF, prediction_BN, prediction_KNN, prediction_LR

# Metodo che applica il post processing
def windowPrediction(list_prediction):

    # Prendo la lunghezza comune per ogni fold
    len_vector = len(list_prediction[0])

    # liste temporanea che conterrà le predizioni per quel determinato minuto in cui andiamo a vedere quale sia la
    # classe prevalente
    list_temp =[]

    # Lista finale che conterrà, per quel determinato minuto, la classe prevalente precedentemene identificata
    list_final =[]

    # Limiti inferiore e superiore della finestra di 1 minuto
    i = 0
    j = 60

    # Indice che identifica la lista sulla quale viene eseguita la conta delle occorrenze della classe prevalente
    idx = 0

    # Finchè non arriviamo alla fine delle predizioni
    while (j <= len_vector):

        # Contatori delle occorrenze per la classe 1 e 3
        n_1 = 0
        n_3 = 0

        # Inseriamo le predizioni all'interno della finestra definita da i e j nella lista temporanea
        list_temp.append(list_prediction[:,i:j])

        # Per ogni riga dentro la lista temporanea, andiamo a contare il numero di 1
        for x in list_temp[idx]:
            x = x.tolist()
            n_1 += x.count(1)

        # Per ogni riga dentro la lista temporanea, andiamo a contare il numero di 3
        for y in list_temp[idx]:
            y = y.tolist()
            n_3 += y.count(3)

        # Controlliamo quale sia la classe prevalente, e una volta identificata la inseriamo nella lista finale
        if(n_1 > n_3):
            if(j!=1170):
                l = np.full((1, 60), 1).tolist()
            else:
                l = np.full((1, 30), 1).tolist()
            list_final += l[0]
        else:
            if (j != 1170):
                l = np.full((1, 60), 3).tolist()
            else:
                l = np.full((1, 30), 3).tolist()
            list_final += l[0]

        # Aggiornamo la finestra passando al minuto successivo
        i = j +1
        j = j+60
        if(j==1200):
            j-=30

        # Prendo in considerazione la lista successiva
        idx +=1

    # Restituisco la lista che conterrà le predizioni più frequenti per ogni minuto considerato
    return list_final




