import time
from pprint import pprint
from pathlib import Path
import pandas as pd 
import matplotlib
import scipy.io as sio
from numpy import double
from scipy import stats
from matplotlib import pyplot as plt
import math
from methods import *
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge,LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import csv
import os.path
import glob

"""
with open('file.csv', 'w+', newline='') as file:
    writer = csv.writer(file)

    allFiles = glob.glob("data\*.csv")
    flag = False

    for file_tmp in allFiles:
        with open(file_tmp, 'r', newline='') as file_tmp:
            csv_reader = csv.reader(file_tmp, delimiter=',')

            for (i,row) in enumerate(csv_reader):
                if (not flag):
                    writer.writerow(row)
                    flag = True
                else:
                    if(i > 0):
                        writer.writerow(row)

        file_tmp.close()
file.close()
"""

"""
IDEA PROGETTO:
Crea un feature vector per ogni file, dunque crea un for che parta dalla lettura del csv
e che arrivi alla generazione del feature vector di quel file. Quando leggi il file csv, siccome ogni
file ha solo una classe, prendi quel valore, e quando stai inserendo Fzlist nel feature vector, fai 
l'append della classe al vettore e scrivilo nel csv. Una volta fatto ciò, ci saranno 80 feature vector,
a questo punto per la fase di regressione, bisogna cercare i metodi per ogni classificatore, del resto non
dovrebbe cambiare nulla.
"""

allFiles = glob.glob("data\*.csv")
print(allFiles)
print(len(allFiles))


if not (os.path.exists("Feature_vector0.csv")):
    for (counter,file_csv) in enumerate(allFiles):

        print(file_csv)
        print(counter)
        list_data = []
        list_class = []

        with open(file_csv, 'r', newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')

            for (i,row) in enumerate(csv_reader):
                if(i>0):
                    array_temp = row
                    array_temp = np.array(array_temp)
                    list_data.append(array_temp[1:22].astype(np.float))
                    list_class.append(array_temp[21].astype(np.float))

            #list_data = list_data.tolist()

            """
            for (i,row) in enumerate(list_data):
        
                classe = round(row[len(row) - 1],0).astype(np.int)
                #list_data[i] = list_data[i].tolist()
                list_data[i][len(row)-1] = classe
        
                #row[len(row)-1] = round(row[len(row)-1],0).astype(np.int)
                #list_data[len(row)-1] = row[len(row)-1]
            """

        file.close()

        #print("Lista dati")
        #print(list_data)
        #print(len(list_data))
        #print("")


        #print("Lista Classi")
        #print(list_class)
        #print(len(list_class))
        list_data = np.array(list_data)

        class_file = list_class[0]
        #print(class_file)
        #time.sleep(10000)

        # Lista che conterrà i valori di ogni canale analizzato
        list_channels = []

        # Per ogni canale considerato, avremo la feature extraction
        feature_vector = []

        # Indice che punta ad un determinato canale
        idx = 0

        # Inizio calcoli

        # Liste contenenti le figure per ogni canale con i rispettivi subplot
        list_figure = []
        list_subplt = []

        i = 0
        while (i < (len(list_data[0])-1)):
            list_channels, list_figure, list_subplt = addChannels(list_channels, i, list_data, idx, list_figure,
                                                                  list_subplt)
            idx += 1
            i += 1

        #pprint(list_channels)

        # Print lunghezza lista canali
        print("Canali contenuti nella lista: " + str(len(list_channels)))

        # Indice dei canali
        idx = 0

        # Liste che conterrano i valori dello spettrogramma (simile al paper) di ogni canale
        Pidx = []
        fidx = []
        bidx = []
        immidx = []

        # Liste che conterrano i valori dello spettrogramma (con parametri) di ogni canale
        Pxx = []
        freqs = []
        xbins = []
        im = []

        sampleFreq = 128

        # Scorriamo tutti i canali
        while (idx < len(list_channels)):
            # Calcoliamo gli spettrogrammi (nelle due versioni) di ogni canale
            P, f, b, imm, P2, f2, b2, imm2 = computeSpecgram(idx, list_channels, sampleFreq, list_subplt)

            # Inserisco i valori dello spettrogramma (paper)
            Pidx.append(P)
            fidx.append(f)
            bidx.append(b)
            immidx.append(imm)

            # Inserisco i valori dello spettrogramma (con parametri)
            Pxx.append(P2)
            freqs.append(f2)
            xbins.append(b2)
            im.append(imm2)

            # Incrememto contatore per analizzare il canale successivo
            idx += 1

        #plt.show()
        plt.close('all')

        #time.sleep(10000)

        # Variabili usate per applicare la finestra di 15s sullo spettrogramma, così da ottenere, per ogni finestra e per
        # ogni canale, un feature vector da 252 elementi

        # k è il limite superiore della finestra
        k = 15 #finesta di 15 secondi
        # finestra di un minuto
        #k = 60

        # j è il limite inferiore della finestra
        j = 0

        # # Lista che conterrà l'average spectral power per ogni banda/gruppo di frequenza considerata
        spec_values = []

        # Finestra di blackman con parametro 1920 = 128 * 15s
        win = np.blackman(32)

        start_time = time.time()
        # Lista contenente i feature vector da 252 elementi
        feature_final = []

        # Indice rappresentate le chiavi per i dizionari
        index = 0

        # Prendo il nome del file csv in base al record specficato ad inizio codice
        name_file_csv = str('Feature_vector'+str(counter)+'.csv')

        """
        print(len(xbins))
        for el in xbins:
            print(len(el))
    
        #time.sleep(1000)
    
        
    
        print("Valore k %s" %k)
        print(len(xbins[0]))
        """

        if (k > len(xbins[0])):
            print("File" + str(name_file_csv))
            k = len(xbins[0]) - 1


        # Se il nome del file specificato non esiste, allora devo creare il csv contenente i feature vector per quel record
        if not (os.path.exists(name_file_csv)):

            # Creo il file csv
            with open(name_file_csv, 'w+', newline='') as file:

                writer = csv.writer(file)

                # Scorriamo la finestra, fino ad arrivare alla fine dello spettrogramma oppure fino a che non raggiungo i 30
                # minuti di osservazione

                # Scorro ogni canale
                while (k < len(xbins[0])):
                    for idx in range(0, len(list_channels)):
                        #print("Canale" + str(idx))
                        # Step 2: Bin 0.5Hz frequency bins
                        # Dal paper:
                        # These were subsequently binned into 0.5 Hz frequency bands by using average, thus, evaluating an average spectral
                        # power in each 0.5 Hz frequency band from 0 to 64 Hz.
                        # # Tramite il metodo psd, a cui passiamo gli stessi parametri dello spettrogramma precedente, andiamo ad ottenere
                        # Power e f, dove Power è un array contenente i values spectral power associati ad ogni frequenza.
                        # Il parametro Pxx[idx][:, j:k] prende per ogni canale idx, tutti i valori delle frequenze (righe) della finestra
                        # definita dagli indici j e k


                        """
                        IDEA: Dato che in questo for scorriamo delle finestre di 15 secondi su tutti i canali, andiamo a prendere
                        da list_data i 15 valori (in base al valore di j e k) di classe associati a quel canale così
                        print(list_data[j:k,len(list_data[idx])-1])
                        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                        Una volta ottenuta questa lista, andiamo a considerare la classe prevalente, ovvero 1. Una volta fatto ciò,
                        inseriamo questo 1 in un'altra nuova lista, che conterrà il valore della classe prevalente (per la finestra
                         considerata) di ogni canale che andremo a considerare. Una volta che questo for termina, avremo una lista finale
                         che avrà 20 valori (20 canali), supponiamo che sia 
                         [1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 1 1 0 0]
                         Una volta usciti dal for, andiamo a vedere quale sia la classe prevalente in questa lista, ovvero 1. 
                         Questo valore sarà la classe che andremo ad appendere a fzlist.
                         
                         Mio dubbio: se prendo i primi 15 valori di ogni canale, ho sempre gli stessi valori di classe per tutti
                         i canali? Da controllare
                        """



                        if(k == 0):
                            Power, f = plt.psd(Pxx[idx][:,0], Fs=sampleFreq, window=win, NFFT=32, pad_to=31)
                        else:
                            Power, f = plt.psd(Pxx[idx][:, j:k], Fs=sampleFreq, window=win, NFFT=32, pad_to=31)
                        # Power, f = ax4.psd(w, Fs=sampleFreq, window=win, NFFT=1920, pad_to=1024)


                        # Richiamo il metodo per calcolare l'average spectral power (all'interno è presente lo step 3)
                        spec_values = average_spectral_power(sampleFreq, f, spec_values, Power)

                        # Step 4
                        # Dal paper:
                        # Finally, the binned and frequency-restricted spectrograms were temporally smoothed by using a 15 s-running average

                        # The running mean is a case of the mathematical operation of convolution. For the running mean, you slide a window
                        # along the input and compute the mean of the window's contents. For discrete 1D signals, convolution is the same thing,
                        # except instead of the mean you compute an arbitrary linear combination, i.e. multiply each element by a corresponding
                        # coefficient and add up the results

                        mov_avg = movingaverage(spec_values, 15)
                        spec_values.clear()


                        # Step 5:
                        # The final feature vector was then formed by converting the power values at each time-point t into decibel form
                        # and combining the spectra from all 7 input EEG channels into a single, joint feature vector, where c enumerates
                        # the input EEG channels.

                        # Applichiamo la formula indicata nel paper, per ogni elemento y contenuto nella lista mov_avg
                        #feature_vector.append([10 * math.log(y, 10) for y in mov_avg])

                        feature_vector.append([10 * math.log(y, 10) for y in mov_avg])

                        spec_values.clear()

                    # Dal paper:
                    # Therefore producing a vector with a dimensionality of 252, characterizing the distribution of the power of the EEG
                    # signal over all EEG channels and the frequencies from 0 to 18 Hz at a 0.5 Hz step
                    # Una volta elaborati i valori contenuti nella finestra definita da j:k, per tutti i canali, definisco il feature
                    # vector finale
                    feature_final = finalFeatureVector(feature_vector)



                    # Scale values with z-scores, so the scores are scaled as if your mean were 0 and standard deviation were 1
                    # Abbiamo applicato zScore in quanto è la funzione corrispondente al parametro "auto-scaling" di Matlab
                    fzScore = stats.zscore(feature_final)



                    # Questa lista contiene i valori del feature vector (scalato) che verrà inserito come riga nel csv, prima di
                    # inserirlo lo converto in lista per poter inserire alla fine la classe specificata
                    fzList = fzScore.tolist()



                    fzList.append(class_file)

                    #print(fzList)
                    # Scrivo la riga nel file csv
                    writer.writerow(fzList)


                    # Pulisco il vettore così da risparmiare memoria
                    feature_vector.clear()

                    # Sposto la finestra di un secondo, e rieseguo i calcoli, così da avere, per ogni finestra, un feature vector
                    # da 252 elementi

                    j = j + 1
                    k = k + 1

                    # Incrememnto indice
                    index += 1

                    #print(k)
                    #print(index)

                    #j = k + 1
                    #k = k + 60
                    idx = 0

            # Una volta arrivato alla fine dello spettrogramma (o a 30 minuti), chiudo il file, ho terminato
            file.close()
            print("--- %s seconds ---" % (time.time() - start_time))

"""

feature_vector_files = glob.glob("Feature_vector*.csv")
list_unified_vector = []
for file in feature_vector_files:
    with open(file, 'r', newline='') as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = list(csv_reader)
        row_count = len(data)
        print(row_count)
        if (row_count == 1):
            print(f)
            for (i,row) in enumerate(data):
                    list_unified_vector.append(row)
            f.close()
            os.remove(f.name)
        else:
            f.close()





print(list_unified_vector)
print(len(list_unified_vector))


with open('Feature_vector_mix.csv', 'w+', newline='') as file:
    writer = csv.writer(file)

    for l in list_unified_vector:
        writer.writerow(l)

    file.close()


"""


# Step Classificatore
# A questo punto, ogni file Feature vector contiene N feature vectors da 252 elementi, uno per ogni
# finestra calcolata sullo spettrogramma

# Definisco il modello con 100 decision trees
rf = RandomForestRegressor(n_estimators=100)

# Creo il Classificatore Gaussiano
gnb = BayesianRidge()

# Creo il Classificatore KNN con valore 5
knn = KNeighborsRegressor(n_neighbors=5)

# Creo il Classificatore
logreg = LinearRegression()

svm_model_linear = LinearSVR(max_iter=1000000)

# Definisco, per ogni classificatore, una lista per ogni metrica considerata, per ogni record, di cui verrà fatta la media
root_square_error_list_SVM_final,square_error_list_SVM_final,r2_list_SVM_final = [],[],[]
root_square_error_list_RF_final,square_error_list_RF_final,r2_list_RF_final = [],[],[]
root_square_error_list_BN_final,square_error_list_BN_final,r2_list_BN_final = [],[],[]
root_square_error_list_KNN_final,square_error_list_KNN_final,r2_list_KNN_final = [],[],[]
root_square_error_list_LR_final,square_error_list_LR_final,r2_list_LR_final = [],[],[]

"""
accuracy_list_RF_final, precision_list_RF_final,recall_list_RF_final, fscore_list_RF_final = [], [],[],[]
accuracy_list_BN_final, precision_list_BN_final, recall_list_BN_final, fscore_list_BN_final  = [], [],[],[]
accuracy_list_KNN_final,precision_list_KNN_final, recall_list_KNN_final, fscore_list_KNN_final  = [],[],[],[]
accuracy_list_LR_final, precision_list_LR_final, recall_list_LR_final, fscore_list_LR_final  = [],[],[],[]
"""
prediction_SVM,prediction_RF,prediction_BN,prediction_KNN,prediction_LR = [],[],[],[],[]

# Contatore per scorrere i record
i = 1

# Liste contenenti le confusion matrix per ogni classificatore, inizialmente inizializzate a 0

"""
cm_final_SVM = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

cm_final_RF = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

cm_final_BN = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

cm_final_KNN = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

cm_final_LR = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]
"""


print("Scelta classificatore:")
print("1 - Kfold con k = 5 applicata su un record alla volta")
print("2 - Kfold con k = 34 applicata su tutti i record")
print("3 - Predizioni")
choice_input = input("Inserisci il numero: ")

if(int(choice_input) == 1):

    print("Kfold con k = 5")
    # Creiamo 5 fold come indicato nel paper. Il parametro shuffle esegue lo shuffle dei dati prima che questi vengano suddivisi
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    # Scorro tutti i record
    while (i < 35):

        # Prendo il nome del record in questione
        files = "Feature_vector_eeg_record" + str(i) + ".csv"
        print(files)

        # Lista contenente le righe che leggo dal file csv
        list_fv = []

        # Apro il file csv specificato in precedenza
        with open(files, 'r', newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')

            # Ogni riga nel file csv viene inserita nella lista
            for row in csv_reader:
                list_fv.append(row)

            # Converto la lista in un array numpy per eseguire operazioni successive
            list_fv = np.array(list_fv)

            # Lista contenente i valori dei feature vector
            feature_values = list_fv[:, 0:251]
            # Lista contenente i valori delle classi
            class_values = list_fv[:, 252]

            # La funzione split() viene richiamata sul campione di dati fornito come argomento.
            # Chiamato ripetutamente, la divisione restituirà ogni gruppo di insieme di training e test. In particolare, vengono
            # restituiti gli array contenenti gli indici nel campione di dati originale di osservazioni da utilizzare per gli insiemi
            # di training e test su ogni iterazione.
            for train_index, test_index in kfold.split(feature_values):
                # Creo l'insieme dei dati usati per il trainig e per il test, specificando gli indici
                X_train, X_test = feature_values[train_index], feature_values[test_index]

                # Creo l'insieme delle label usate per il trainig e per il test, specificando gli indici
                Y_train, Y_test = class_values[train_index], class_values[test_index]

                # Converto i valori in float per le operazioni successive
                X_train = X_train.astype(np.float)
                X_test = X_test.astype(np.float)
                Y_test = Y_test.astype(np.float)
                Y_train = Y_train.astype(np.float)

                # Applico i classificatori sugli insiemi appena definiti, ciascun metodo restituisce l'accuratezza, la precision, la recall e fscore di una fold applicata
                accuracy_list_SVM_final, precision_list_SVM_final, recall_list_SVM_final, fscore_list_SVM_final, cm_final_SVM, prediction_SVM = \
                    computeSVM(X_train, X_test, Y_train, Y_test, accuracy_list_SVM_final, precision_list_SVM_final,
                               recall_list_SVM_final, fscore_list_SVM_final, svm_model_linear,cm_final_SVM,prediction_SVM)

                # Applico gli altri 4 classificatori rimanenti, specificati dal parametro finale

                accuracy_list_RF_final, precision_list_RF_final, recall_list_RF_final, fscore_list_RF_final, cm_final_RF, prediction_RF = \
                    computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_RF_final, precision_list_RF_final,
                                      recall_list_RF_final, fscore_list_RF_final, rf, cm_final_RF, prediction_RF)

                accuracy_list_BN_final, precision_list_BN_final, recall_list_BN_final, fscore_list_BN_final, cm_final_BN, prediction_BN = \
                    computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_BN_final, precision_list_BN_final,
                                      recall_list_BN_final, fscore_list_BN_final, gnb, cm_final_BN, prediction_BN)

                accuracy_list_KNN_final, precision_list_KNN_final, recall_list_KNN_final, fscore_list_KNN_final, cm_final_KNN, prediction_KNN = \
                    computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_KNN_final,
                                      precision_list_KNN_final,
                                      recall_list_KNN_final, fscore_list_KNN_final, knn, cm_final_KNN, prediction_KNN)

                accuracy_list_LR_final, precision_list_LR_final, recall_list_LR_final, fscore_list_LR_final, cm_final_LR, prediction_LR = \
                    computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_LR_final, precision_list_LR_final,
                                      recall_list_LR_final, fscore_list_LR_final, logreg, cm_final_LR, prediction_LR)

        # Chiudo il file csv
        file.close()
        # Passo al record successivo
        i += 1

elif(int(choice_input) == 2):



    # Creiamo 34 fold su tutti i dati
    allFiles = glob.glob("Feature*.csv")
    print("Kfold con k = %s" %len(allFiles))
    kfold = KFold(len(allFiles), n_splits=len(allFiles), shuffle=False)


    # La funzione split() viene richiamata sul campione di dati fornito come argomento.
    # Chiamato ripetutamente, la divisione restituirà ogni gruppo di insieme di training e test. In particolare, vengono
    # restituiti gli array contenenti gli indici nel campione di dati originale di osservazioni da utilizzare per gli insiemi
    # di training e test su ogni iterazione. In questo caso lavoriamo su tutti i file, dunque dataTrain conterrà per ogni fold
    # i dati di 33 file mentre dataTest conterrà i dati di un solo record
    count = 1
    for train_files, cv_files in kfold.split(allFiles):

        dataTrain = pd.concat((pd.read_csv(allFiles[idTrain], header=None) for idTrain in train_files))
        dataTest = pd.concat((pd.read_csv(allFiles[idTest], header=None) for idTest in cv_files))

        """
        print(len(dataTrain))
        pprint(dataTrain)
        print(len(dataTest))
        pprint(dataTest)
        """



        # Lista contenente i valori dei feature vector
        X_train, Y_train = dataTrain[np.arange(0,300)], dataTrain[[300]]
        # Lista contenente i valori delle classi
        X_test, Y_test = dataTest[np.arange(0,300)], dataTest[[300]]

        Y_train = np.ravel(Y_train)
        Y_test = np.ravel(Y_test)

        """
        print("X Train")
        print(len(X_train))
        pprint(X_train)
        print("")

        print("X Test")
        print(len(X_test))
        pprint(X_test)
        print("")

        print("Y Train")
        print(len(Y_train))
        pprint(Y_train)
        print("")

        print("Y Test")
        print(len(Y_test))
        pprint(Y_test)
        print("")

        time.sleep(10000)
        """

        print("SVM")
        # Applico i classificatori sugli insiemi appena definiti, ciascun metodo restituisce l'accuratezza, la precision, la recall e fscore di una fold applicata
        root_square_error_list_SVM_final, square_error_list_SVM_final, r2_list_SVM_final, prediction_SVM = \
            computeSVM(X_train, X_test, Y_train, Y_test, root_square_error_list_SVM_final, square_error_list_SVM_final,
                       r2_list_SVM_final, svm_model_linear,prediction_SVM)

        # Applico gli altri 4 classificatori rimanenti, specificati dal parametro finale
        print("RF")
        root_square_error_list_RF_final, square_error_list_RF_final, r2_list_RF_final, prediction_RF = \
            computeClassifier(X_train, X_test, Y_train, Y_test, root_square_error_list_RF_final, square_error_list_RF_final,
                              r2_list_RF_final, rf,prediction_RF)
        print("BN")
        root_square_error_list_BN_final, square_error_list_BN_final, r2_list_BN_final, prediction_BN = \
            computeClassifier(X_train, X_test, Y_train, Y_test, root_square_error_list_BN_final, square_error_list_BN_final,
                              r2_list_BN_final, gnb, prediction_BN)
        print("KNN")
        root_square_error_list_KNN_final, square_error_list_KNN_final, r2_list_KNN_final, prediction_KNN = \
            computeClassifier(X_train, X_test, Y_train, Y_test, root_square_error_list_KNN_final, square_error_list_KNN_final,
                              r2_list_KNN_final, knn, prediction_KNN)


        print("LR")
        root_square_error_list_LR_final, square_error_list_LR_final, r2_list_LR_final, prediction_LR = \
            computeClassifier(X_train, X_test, Y_train, Y_test, root_square_error_list_LR_final, square_error_list_LR_final,
                              r2_list_LR_final, logreg, prediction_LR)


        print(count)

        count += 1

    print("SVM")
    printAverageValues(root_square_error_list_SVM_final, square_error_list_SVM_final, r2_list_SVM_final)
    print("")

    print("Random Forest")
    printAverageValues(root_square_error_list_RF_final, square_error_list_RF_final, r2_list_RF_final)
    print("")

    print("BN")
    printAverageValues(root_square_error_list_BN_final, square_error_list_BN_final, r2_list_BN_final)
    print("")

    print("KNN")
    printAverageValues(root_square_error_list_KNN_final, square_error_list_KNN_final, r2_list_KNN_final)
    print("")


    print("LR")
    printAverageValues(root_square_error_list_LR_final, square_error_list_LR_final, r2_list_LR_final)
    print("")


    exit(1)


    # FASE POST-PROCESSING
    # Una volta calcolate le predizioni per ciascun classificatore (ciascuna predizione è inserita in una rispettiva lista)
    # andiamo a saalvarci queste predizioni su un file CSV

    if not (os.path.exists("Predictions.csv")):
        writePrediction(prediction_SVM,prediction_RF,prediction_BN,prediction_KNN,prediction_LR)

    # Una volta scritto il file delle predizioni, salviamo il loro valore all'interno delle liste
    prediction_SVM,prediction_RF, prediction_BN, prediction_KNN, prediction_LR = readPrediction()

    # Applico la finestra di 1 minuto (non sovrapposta) sulle predizioni di ogni classificatore
    list_SVM_P = windowPrediction(prediction_SVM)
    list_RF_P = windowPrediction(prediction_RF)
    list_BN_P = windowPrediction(prediction_BN)
    list_KNN_P = windowPrediction(prediction_KNN)
    list_LR_P = windowPrediction(prediction_LR)

    # Convertiamo in int i valori di Y_test che verrà usato per calcolare le metriche
    Y_test = np.array(Y_test).astype(np.int)

    # Per ogni classificatore, calcoliamo le metriche in base alle nuove predizioni
    accSVM, precSVM, recallSVM, fscoreSVM, cmSVM = metricsPrevision(Y_test, list_SVM_P)
    accRF, precRF, recallRF, fscoreRF, cmRF = metricsPrevision(Y_test, list_RF_P)
    accBN, precBN, recallBN, fscoreBN, cmBN = metricsPrevision(Y_test, list_BN_P)
    accKNN, precKNN, recallKNN, fscoreKNN, cmKNN = metricsPrevision(Y_test, list_KNN_P)
    accLR, precLR, recallLR, fscoreLR, cmLR = metricsPrevision(Y_test, list_LR_P)


    # Stampo le metriche per ogni classificatore

    print("SVM")
    printValues(accSVM, precSVM, recallSVM, fscoreSVM)

    print("RF")
    printValues(accRF, precRF, recallRF, fscoreRF)

    print("BN")
    printValues(accBN, precBN, recallBN, fscoreBN)

    print("KNN")
    printValues(accKNN, precKNN, recallKNN, fscoreKNN)

    print("LR")
    printValues(accLR, precLR, recallLR, fscoreLR)




"""
prediction_SVM = np.array(prediction_SVM)
print(prediction_SVM.shape)
print(prediction_SVM)
svm = prediction_SVM[:, 0:5]
print(svm)

# Stampo le matrici di confusione complessive per ogni classificatore
print("Matrice finale SVM")
print(cm_final_SVM)
print("")

print("Matrice finale Rf")
print(cm_final_RF)
print("")

print("Matrice finale BN")
print(cm_final_BN)
print("")

print("Matrice finale KNN")
print(cm_final_KNN)
print("")

print("Matrice finale LR")
print(cm_final_LR)
print("")

# Una volta uscito dal while, ho terminato di analizzare tutti i record, dunque per ogni classificatore, calcolo
# l'accuratezza, la precision, la recall e fscore media, in base ai valori nelle liste ottenuti da ogni record

print("SVM")
printAverageValues(accuracy_list_SVM_final, precision_list_SVM_final, recall_list_SVM_final, fscore_list_SVM_final)
print("")

print("Random Forest")
printAverageValues(accuracy_list_RF_final, precision_list_RF_final, recall_list_RF_final, fscore_list_RF_final)
print("")

print("BN")
printAverageValues(accuracy_list_BN_final, precision_list_BN_final, recall_list_BN_final, fscore_list_BN_final)
print("")

print("KNN")
printAverageValues(accuracy_list_KNN_final, precision_list_KNN_final, recall_list_KNN_final, fscore_list_KNN_final)
print("")

print("LR")
printAverageValues(accuracy_list_LR_final, precision_list_LR_final, recall_list_LR_final, fscore_list_LR_final)
print("")

"""