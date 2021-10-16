import time
import pandas as pd
import numpy as np
import matplotlib
from scipy.integrate import simps
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


# Dal dataFrame, prendiamo i valori del canale "col", li inseriamo in listchannel e plottiamo i valori
def addChannels(list_channels, col, table_edit,idx,list_figure, list_subplt):
    print(col)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)

    list_figure.append(fig)
    list_subplt.append((ax1, ax2, ax3))

    # Lista dei valori dei canali
    list_channels.append(table_edit[col].values.tolist())

    # Rappresento graficamente i valori nella mia lista, ovvero le frequenze associate ai campioni
    ax1.plot(list_channels[idx])
    ax1.title.set_text('Signal ' + str(col))
    ax1.set_xlabel('nSamples')
    ax1.set_ylabel('Frequency [Hz]')

    return list_channels, list_figure, list_subplt


def computeSpecgram(idx, list_channels,sampleFreq, list_subplt):

    ax1,ax2,ax3 = list_subplt[idx]
    # Trasformo i valori in x da secondi a minuti
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M', time.gmtime(ms)))
    ax2.xaxis.set_major_formatter(formatter)
    ax3.xaxis.set_major_formatter(formatter)

    # Se viene calcolato lo spettrogramma del segnale specificando solamente la frequenza di campionamento, si ottiene uno
    # spettrogramma abbastanza simile a quello mostrato in fig 6 nel paper. L'intuizione è che gli autori del paper abbiano
    # mostrato dunque lo spettrogramma del segnale senza specificare alcun tipo di parametro
    P, f, b, imm = ax2.specgram(list_channels[idx], Fs=sampleFreq, cmap='jet')
    #fig.colorbar(imm, ax=ax2).set_label('Intensity [dB]')
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set(xlabel='Time [min]', ylabel='Frequency [Hz]')

    # Primo metodo con cui genero lo spettrogramma, uso la finestra di Blackman con dimensione pari a 1920, ma di questa
    # finestra considero solamente 1024 punti tramite pad_to, come specificato nel paper. Il parametro noverlap
    # è stato inserito per avere degli step di 1 secondo tra un valore e l'altro
    Pxx, freqs, xbins, im = ax3.specgram(list_channels[idx], Fs=sampleFreq, window=np.blackman(1920), NFFT=1920,
                                         pad_to=1024, noverlap=sampleFreq * 14,
                                         cmap='jet')
    ax3.set_title('Spectogram')
    ax3.set(xlabel='Time [min]', ylabel='Frequency [Hz]')

    # Aggiunta colorbar per indicare l'intensità
    #fig.colorbar(im, ax=ax3).set_label('Intensity [dB]')

    # restituisco i valori
    return P, f, b, imm, Pxx, freqs, xbins, im



def average_spectral_power(sampleFreq, f, spec_values, Power):

    # Risoluzione di frequenza
    freq_res = sampleFreq / 1024

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

    # Scorriamo tutte le frequenze
    while (i < len(f)):
        # Se la frequenza considerata non supera il limite superiore della banda
        if (f[i] <= high):
            # Andiamo a settare a True tutte le frequenze che ricando dentro il range specificato
            idx_delta.append(np.logical_and(f > low, f <= high))
            # Dal momento che dobbiamo calcolare l'average spectral power di un intervallo di frequenze, è necessario
            # usare degli integrali, utilizziamo dunque il metodo simps (Simpsons' rule)
            avg_sp = simps(Power[idx_delta[j]], dx=freq_res)
            spec_values.append(avg_sp)

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
def computeSVM(X_train_df,X_test_df,Y_train,Y_test,accuracy_list,precision_list,recall_list,fscore_list,svm_model_linear,cf_final):

    # Specifiamo la kernel function, in questo caso 'linear' (citata nel paper) per costruire il modello su
    # gli insiemi di training creati in precedenza
    #svm_model_linear = SVC(kernel='linear').fit(X_train_df, Y_train)
    #svm_model_linear = LinearSVC(max_iter=100000).fit(X_train_df, Y_train)
    svm_model_linear.fit(X_train_df, Y_train)

    predictions = svm_model_linear.predict(X_test_df)

    # Applico il modello sugli insiemi di test e score restituisce l'accuratezza rispetto agli insiemi dati
    # l'accuratezza della fold corrente viene inserita in una lista
    #accuracy_list.append(svm_model_linear.score(X_test_df, Y_test))
    accuracy_list.append(metrics.accuracy_score(Y_test, predictions))

    # Il metodo precision_recall_fscore_support restituisce le metriche di interesse
    precision, recall, fscore, n = precision_recall_fscore_support(Y_test,predictions, average=None)

    # Inserisco le metriche nelle rispettive liste
    precision_list.append(precision)
    recall_list.append(recall)
    fscore_list.append(fscore)

    # Calcolo confusion matrix della fold corrente
    cf_temp = confusion_matrix(Y_test, predictions)

    # Sommo la confusion matrix appena calcolata con quella finale
    cf_final = cf_final + cf_temp

    return accuracy_list, precision_list,recall_list,fscore_list, cf_final

# Applico il classificatore definito dal parametro "classifier"
def computeClassifier(X_train_df,X_test_df,Y_train,Y_test,accuracy_list,precision_list,recall_list,fscore_list,classifier, cf_final):

    # Alleno il modello con i dati di training
    classifier.fit(X_train_df.astype(np.float), Y_train.astype(np.float))

    # Uso il classificatore per le predizioni sui dati di test
    predictions = classifier.predict(X_test_df.astype(np.float))

    # Calcolo accuratezza del classificatore in base alle predizioni e ai valori di classe di test
    accuracy_list.append(metrics.accuracy_score(Y_test.astype(np.float), predictions))

    # Il metodo precision_recall_fscore_support restituisce le metriche di interesse
    precision, recall, fscore, n = precision_recall_fscore_support(Y_test.astype(np.float), predictions, average=None)

    # Inserisco le metriche nelle rispettive liste
    precision_list.append(precision)
    recall_list.append(recall)
    fscore_list.append(fscore)

    # Calcolo confusion matrix della fold corrente
    cf_temp = confusion_matrix(Y_test.astype(np.float), predictions)

    # Sommo la confusion matrix appena calcolata con quella finale
    cf_final = cf_final + cf_temp

    # Restituisco la lista contenente le accurarezze per ciascuna fold
    return accuracy_list, precision_list,recall_list,fscore_list,cf_final


# Restituisco accuratezza media della lista che viene passata come parametro

def getAverage(accuracy_list,precision_list,recall_list,fscore_list,b):

    # b è un booleano che specifica se è necessario calcolare la percentuale o meno
    if(b==True):
        accuracy = round((np.mean(accuracy_list) * 100),2)
        precision = round((np.mean(precision_list) * 100), 2)
        recall = round((np.mean(recall_list) * 100), 2)
        fscore = round((np.mean(fscore_list) * 100), 2)
    else:
        accuracy = round(np.mean(accuracy_list), 2)
        precision = round(np.mean(precision_list), 2)
        recall = round(np.mean(recall_list), 2)
        fscore = round(np.mean(fscore_list), 2)


    # Restituisco accuratezza, precision, recall e fscore media
    return accuracy,precision, recall,fscore


# Passando come parametri le liste delle metriche, calcolo la media e stampo i valori

def printAverageValues(accuracy_list_final,precision_list_final,recall_list_final,fscore_list_final):

    a, p, r, fs = getAverage(accuracy_list_final, precision_list_final, recall_list_final,
                                          fscore_list_final, True)


    print("Media accuratezza records")
    print(str(a) + "%")
    print("")

    print("Media precision records")
    print(str(p) + "%")
    print("")

    print("Media recall records")
    print(str(r) + "%")
    print("")

    print("Media fscore records")
    print(str(fs) + "%")
    print("")
