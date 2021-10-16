import scipy.io as sio
from scipy import stats
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from methods import *
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
import csv
import os.path
import glob


# Carico il file matlab tramite loadmat che mi restiuisce un dizionario
n_file = "eeg_record1"
mat = sio.loadmat("EEG Data/" + str(n_file) + ".mat")

# Abbiamo bisogno solamente dei valori associati alla chiave 'o'
raw_data = mat['o']

# Timestamp sarà un dataframe contenente le label di classe in base al minuto che viene considerato
timestamp = definingClass(raw_data)

# Prendo le informazioni necessaria, come la frequenza di campionamento e il numero di campioni
sampleFreq = raw_data[0, 0]['sampFreq'].flat[0]
numSamples = raw_data[0, 0]['nS'].flat[0]

"""
Colonne

0-'EDCOUNTER' 1-'EDINTERPOLATED' 2-'EDRAWCQ'
3-'EDAF3' 4-'EDF7'
5-'EDF3' 6-'EDFC5'
7-'EDT7' 8-'EDP7'
9-'EDO1' 10-'EDO2'
11-'EDP8' 12-'EDT8'
13-'EDFC6' 14-'EDF4'
15-'EDF8' 16-'EDAF4'
17-'EDGYROX' 18-'EDGYROY'
19-'EDTIMESTAMP' 20-'EDESTIMESTAMP'
21-'EDFUNCID' 22-'EDFUNCVALUE'
23-'EDMARKER' 24'EDSYNCSIGNAL'
"""

# Mi creo un dizionario in cui gestisco meglio i dati per lavoraci sopra, genero quindi prima le chiavi
keys = range(len(raw_data["data"][0, 0]))

# Genero il dizionario, zip è un metodo che associa ad ogni elemento in keys la rispettiva riga in raw_data
dict_data = dict(zip(keys, raw_data["data"][0, 0]))

# Genero il dataframe completo, orient = index serve per ordinare i valori delle righe usando le chiavi del dizionario,
# mentre columns associa un'etichetta ad ogni colonna
table_data = pd.DataFrame.from_dict(dict_data, orient='index', columns=
['COUNTER', 'INTERPOLATED', 'RAWCQ', 'CHANNEL AF3', 'CHANNEL F7', 'CHANNEL F3', 'CHANNEL FC5', 'CHANNEL T7',
 'CHANNEL P7',
 'CHANNEL O1', 'CHANNEL O2', 'CHANNEL P8', 'CHANNEL T8', 'CHANNEL FC6', 'CHANNEL F4', 'CHANNEL F8', 'CHANNEL AF4',
 'GYROX', 'GYROY', 'TIMESTAMP', 'ESTIMESTAMP', 'FUNCID', 'FUNCVALUE', 'MARKER', 'SYNCSIGNAL'])

# Impostazioni di pandas per mostrare il dataframe con le colonne complete
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

# Da quanto scritto su kaggle, i dati EEG sono contenuti nei canali 3:16, di questi però ci servono solo:
# 4 - EDF7, 5 - EDF3, 7 - EDT7, 8 - EDP7, 9 - EDO1, 10 - EDO2, 11 - EDP8, 12 - EDT8, 16 - EDAF4
# A quanto scritto, i canali 7 (EDT7) e 12 (EDT8) sono inutili, in quanto solo canali di referenza
# Aggiungiamo, come nel paper, anche le colonne X e Y, ovvero la 17 e 18

# Modifico la tabella precedente andando a selezionare solamente le colonne che ci servono
# I canali T7 e T8 sono inutili in quanto sono canali di referenza
table_edit = table_data[['COUNTER', 'INTERPOLATED', 'CHANNEL F7', 'CHANNEL F3', 'CHANNEL P7',
                         'CHANNEL O1', 'CHANNEL O2', 'CHANNEL P8', 'CHANNEL AF4', 'GYROX', 'GYROY']]

# Aggiungo la colonna delle classi all'altro dataframe
combined_df = pd.concat([table_edit, timestamp[['Class']]], axis=1)

# Lista che conterrà i valori di ogni canale analizzato
list_channels = []

# Per ogni canale considerato, avremo la feature extraction
feature_vector = []

# Indice che punta ad un determinato canale
idx = 0

# Liste contenenti le figure per ogni canale con i rispettivi subplot
list_figure = []
list_subplt = []

# Otteniamo i valori di ogni canale
for col in table_edit.columns:
    if (col == 'CHANNEL F7' or col == 'CHANNEL F3' or col == 'CHANNEL P7'
            or col == 'CHANNEL O1' or col == 'CHANNEL O2' or col == 'CHANNEL P8' or col == 'CHANNEL AF4'):
        # list_channels = addChannels(list_channels,col,table_edit,ax1,idx,ax2)
        list_channels, list_figure, list_subplt = addChannels(list_channels, col, table_edit, idx, list_figure,
                                                              list_subplt)
        idx += 1

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

plt.show()

# Variabili usate per applicare la finestra di 15s sullo spettrogramma, così da ottenere, per ogni finestra e per
# ogni canale, un feature vector da 252 elementi

# k è il limite superiore della finestra
k = 15

# j è il limite inferiore della finestra
j = 0

# # Lista che conterrà l'average spectral power per ogni banda/gruppo di frequenza considerata
spec_values = []

# Finestra di blackman con parametro 1920 = 128 * 15s
win = np.blackman(1920)

start_time = time.time()

# Lista contenente i feature vector da 252 elementi
feature_final = []

# Indice rappresentate le chiavi per i dizionari
index = 0

# Lunghezza spettrogramma (xbins[0] è uguale per tutti i canali)
len_spec = len(xbins[0])

# Prendo il nome del file csv in base al record specficato ad inizio codice
name_file_csv = str('Feature_vector_' + str(n_file) + '.csv')

# Se il nome del file specificato non esiste, allora devo creare il csv contenente i feature vector per quel record
if not (os.path.exists(name_file_csv)):

    # Creo il file csv
    with open('Feature_vector_' + str(n_file) + '.csv', 'w+', newline='') as file:

        writer = csv.writer(file)
        cont = 0

        # Scorriamo la finestra, fino ad arrivare alla fine dello spettrogramma oppure fino a che non raggiungo i 30
        # minuti di osservazione
        while ((k < len_spec) and (k <= 1800)):
            # Scorro ogni canale
            for idx in range(0, len(list_channels)):
                # Step 2: Bin 0.5Hz frequency bins
                # Dal paper:
                # These were subsequently binned into 0.5 Hz frequency bands by using average, thus, evaluating an average spectral
                # power in each 0.5 Hz frequency band from 0 to 64 Hz.
                # # Tramite il metodo psd, a cui passiamo gli stessi parametri dello spettrogramma precedente, andiamo ad ottenere
                # Power e f, dove Power è un array contenente i values spectral power associati ad ogni frequenza.
                # Il parametro Pxx[idx][:, j:k] prende per ogni canale idx, tutti i valori delle frequenze (righe) della finestra
                # definita dagli indici j e k

                Power, f = plt.psd(Pxx[idx][:, j:k], Fs=sampleFreq, window=win, NFFT=1920, pad_to=1024)

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

            # Qui, in base al tempo, assegnamo la classe opportuna

            if (k <= 600):
                fzList.append(1)
            elif ((j > 600) and (k <= 1200)):
                fzList.append(2)
            elif (j>1200):
                fzList.append(3)
            else:
                if(cont<=7):
                    fzList.append(1)
                elif(cont >7 and cont<=22):
                    fzList.append(2)
                else:
                    fzList.append(3)
                cont+=1


            # Scrivo la riga nel file csv
            writer.writerow(fzList)

            # Incrememnto indice
            index += 1

            # Pulisco il vettore così da risparmiare memoria
            feature_vector.clear()

            # Sposto la finestra di un secondo, e rieseguo i calcoli, così da avere, per ogni finestra, un feature vector
            # da 252 elementi

            j = j + 1
            k = k + 1

            idx = 0


    # Una volta arrivato alla fine dello spettrogramma (o a 30 minuti), chiudo il file, ho terminato
    file.close()
    print("--- %s seconds ---" % (time.time() - start_time))


# Step Classificatore
# A questo punto, ogni file Feature vector contiene N feature vectors da 252 elementi, uno per ogni
# finestra calcolata sullo spettrogramma

# Definisco il modello con 100 decision trees
rf = RandomForestClassifier(n_estimators=100)

# Creo il Classificatore Gaussiano
gnb = GaussianNB()

# Creo il Classificatore KNN con valore 5
knn = KNeighborsClassifier(n_neighbors=5)

# Creo il LogisticRegression
logreg = LogisticRegression(max_iter=10000)

# Creo SVM con kernel lineare
svm_model_linear = LinearSVC(max_iter=100000)


# Definisco, per ogni classificatore, una lista per ogni metrica considerata, per ogni record, di cui verrà fatta la media

accuracy_list_SVM_final,precision_list_SVM_final,recall_list_SVM_final,fscore_list_SVM_final = [],[],[],[]
accuracy_list_RF_final, precision_list_RF_final,recall_list_RF_final, fscore_list_RF_final = [], [],[],[]
accuracy_list_BN_final, precision_list_BN_final, recall_list_BN_final, fscore_list_BN_final  = [], [],[],[]
accuracy_list_KNN_final,precision_list_KNN_final, recall_list_KNN_final, fscore_list_KNN_final  = [],[],[],[]
accuracy_list_LR_final, precision_list_LR_final, recall_list_LR_final, fscore_list_LR_final  = [],[],[],[]


# Contatore per scorrere i record
i = 1

# Liste contenenti le confusion matrix per ogni classificatore, inizialmente inizializzate a 0


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


print("Scelta classificatore:")
print("1 - Kfold con k = 5 applicata su un record alla volta")
print("2 - Kfold con k = 34 applicata su tutti i record")
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
                accuracy_list_SVM_final, precision_list_SVM_final, recall_list_SVM_final, fscore_list_SVM_final, cm_final_SVM = \
                    computeSVM(X_train, X_test, Y_train, Y_test, accuracy_list_SVM_final, precision_list_SVM_final,
                               recall_list_SVM_final, fscore_list_SVM_final, svm_model_linear,cm_final_SVM)

                # Applico gli altri 4 classificatori rimanenti, specificati dal parametro finale

                accuracy_list_RF_final, precision_list_RF_final, recall_list_RF_final, fscore_list_RF_final, cm_final_RF = \
                    computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_RF_final, precision_list_RF_final,
                                      recall_list_RF_final, fscore_list_RF_final, rf, cm_final_RF)

                accuracy_list_BN_final, precision_list_BN_final, recall_list_BN_final, fscore_list_BN_final, cm_final_BN = \
                    computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_BN_final, precision_list_BN_final,
                                      recall_list_BN_final, fscore_list_BN_final, gnb, cm_final_BN)

                accuracy_list_KNN_final, precision_list_KNN_final, recall_list_KNN_final, fscore_list_KNN_final, cm_final_KNN = \
                    computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_KNN_final,
                                      precision_list_KNN_final,
                                      recall_list_KNN_final, fscore_list_KNN_final, knn, cm_final_KNN)

                accuracy_list_LR_final, precision_list_LR_final, recall_list_LR_final, fscore_list_LR_final, cm_final_LR = \
                    computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_LR_final, precision_list_LR_final,
                                      recall_list_LR_final, fscore_list_LR_final, logreg, cm_final_LR)

        # Chiudo il file csv
        file.close()
        # Passo al record successivo
        i += 1

elif(int(choice_input) == 2):

    print("Kfold con k = 34")

    # Creiamo 34 fold su tutti i dati
    allFiles = glob.glob("Feature*.csv")
    kfold = KFold(len(allFiles), n_splits=34, shuffle=False)

    # La funzione split() viene richiamata sul campione di dati fornito come argomento.
    # Chiamato ripetutamente, la divisione restituirà ogni gruppo di insieme di training e test. In particolare, vengono
    # restituiti gli array contenenti gli indici nel campione di dati originale di osservazioni da utilizzare per gli insiemi
    # di training e test su ogni iterazione. In questo caso lavoriamo su tutti i file, dunque dataTrain conterrà per ogni fold
    # i dati di 33 file mentre dataTest conterrà i dati di un solo record
    count = 1
    for train_files, cv_files in kfold.split(allFiles):

        dataTrain = pd.concat((pd.read_csv(allFiles[idTrain], header=None) for idTrain in train_files))
        dataTest = pd.concat((pd.read_csv(allFiles[idTest], header=None) for idTest in cv_files))

        # Lista contenente i valori dei feature vector
        X_train, Y_train = dataTrain[np.arange(0,252)], dataTrain[[252]]
        # Lista contenente i valori delle classi
        X_test, Y_test = dataTest[np.arange(0,252)], dataTest[[252]]

        Y_train = np.ravel(Y_train)
        Y_test = np.ravel(Y_test)


        print("SVM")
        # Applico i classificatori sugli insiemi appena definiti, ciascun metodo restituisce l'accuratezza, la precision, la recall e fscore di una fold applicata
        accuracy_list_SVM_final, precision_list_SVM_final, recall_list_SVM_final, fscore_list_SVM_final,cm_final_SVM = \
            computeSVM(X_train, X_test, Y_train, Y_test, accuracy_list_SVM_final, precision_list_SVM_final,
                       recall_list_SVM_final, fscore_list_SVM_final, svm_model_linear,cm_final_SVM)

        # Applico gli altri 4 classificatori rimanenti, specificati dal parametro finale
        print("RF")
        accuracy_list_RF_final, precision_list_RF_final, recall_list_RF_final, fscore_list_RF_final, cm_final_RF = \
            computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_RF_final, precision_list_RF_final,
                              recall_list_RF_final, fscore_list_RF_final, rf, cm_final_RF)
        print("BN")
        accuracy_list_BN_final, precision_list_BN_final, recall_list_BN_final, fscore_list_BN_final, cm_final_BN = \
            computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_BN_final, precision_list_BN_final,
                              recall_list_BN_final, fscore_list_BN_final, gnb, cm_final_BN)
        print("KNN")
        accuracy_list_KNN_final, precision_list_KNN_final, recall_list_KNN_final, fscore_list_KNN_final, cm_final_KNN = \
            computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_KNN_final, precision_list_KNN_final,
                              recall_list_KNN_final, fscore_list_KNN_final, knn, cm_final_KNN)
        print("LR")
        accuracy_list_LR_final, precision_list_LR_final, recall_list_LR_final, fscore_list_LR_final, cm_final_LR = \
            computeClassifier(X_train, X_test, Y_train, Y_test, accuracy_list_LR_final, precision_list_LR_final,
                              recall_list_LR_final, fscore_list_LR_final, logreg, cm_final_LR)


        print(count)

        count += 1

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

