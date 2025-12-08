# Team-8---Progetto-Machine-Learning-A.A-25-26

# Passi per la configurazione del progetto:
   1. Clonare la repo
   2. creare virtual environment con python 3.13 ```python3.13 -m venv venv```
   3. installare dipendenze e librerie con requirements.txt ```pip install -r requirements.txt```
   4. creare file .env ed aggiungere API KEY di Kaggle. Per ottenerla devi loggare su Kaggle cliccare sull'icona del profilo in alto a destra, poi andare in "settings", scrollare fino ad API Tokens e generarlo.
   5. Incollare il token nel file .env
   6. scaricare il dataset: eseguire dalla root del progetto il comando: ```python utils/dataset_download.py```, il dataset verrà scaricato nella cartella ```data/german-traffic-sign```
   7. Ora puoi trainare i modelli: creali nella cartella ```classifiers```
   8. ENJOY LITTLE MOTHERFUCKER

## Versione di python: 3.13.7
## ‼️IMPORTANTE - Sono state estratte le features nei seguenti file: ```data/features/features_test.npz``` e ```data/features/features_train.npz``` 
Le classi estratte sono ordinate e solo in formato numerico, è stato aggiunto il file ``` utils/gtsrb_classes.py``` che contiene il dizionario con le classi in formato testuale.
Quindi questo prevede che deve essere importato il dizionario quando si vogliono leggere le classi testuali. All'interno del file è presente anche un esempio per la conversione
In ```classifiers/esempio.py``` è presente un altro esempio con l'import del modulo con il dizionario.

## ‼️IMPORTANTE - FARE LO SHUFFLE DELLE FEATURES QUANDO SI TRAINANO I MODELLI

# 💻 Pipeline
### Transfer Learning:
1. Rete pre trained: ResNet 50 / VGG16 - Fatto(resnet)✅
2. Freeze layer finali - Fatto✅
3. Data Pipeline: - Fatto✅
   - Resize Immagini: Ridimensionare le immagini GTSRB (anche se sono 32x32) a 224x224 (o alla dimensione di input richiesta dalla rete scelta).
   - Preprocessing: Usare la stessa normalizzazione usata durante l'addestramento originale su ImageNet. Le librerie (come torchvision) offrono funzioni preprocess_input specifiche per ogni modello. Rifarsi ai lab del professore.
4. Estrazione delle Features(lavorare in batch): - Fatto✅
   - Feed-forward (no back propagation)
   - Flattening
   - Salvataggio label

5. Addestramento dei seguenti classificatori - Da fare❌
   Prima fare PCA, poi allenare modelli:
   - SVM
   - Random Forest
   - KNN
   - MLP
   - Regressore Lineare

6. Inferenza e Valutazione: - Da fare❌
   - Prendi la nuova immagine -> Resize a 224x224.
   - Passala nella CNN (Backbone) -> Ottieni il vettore di feature.
   - Applica la stessa trasformazione PCA usata nel training.
   - Passa il vettore ridotto al classificatore ML
   - Ottieni la classe finale.
7. Metriche: - Da fare❌
   - Accuracy
   - F1-score
   - Inference time
   - model size
   - Robustezza
  
8. Creazione dataset esterno: - Da fare❌
   - Acquisizone foto
   - Etichettatura tramite ROBOFLOW
   - Passare dati al modello e fare inferenza.
  


