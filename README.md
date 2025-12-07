# Team-8---Progetto-Machine-Learning-A.A-25-26


# 💻 Pipeline
### Transfer Learning:
1. Rete pre trained: ResNet 50 / VGG16
2. Freeze layer finali
3. Data Pipeline:
   - Resize Immagini: Ridimensionare le immagini GTSRB (anche se sono 32x32) a 224x224 (o alla dimensione di input richiesta dalla rete scelta).
   - Preprocessing: Usare la stessa normalizzazione usata durante l'addestramento originale su ImageNet. Le librerie (come torchvision) offrono funzioni preprocess_input specifiche per ogni modello. Rifarsi ai lab del professore.
4. Estrazione delle Features(lavorare in batch):
   - Feed-forward (no back propagation)
   - Flattening
   - Salvataggio label
6. Addestramento dei seguenti classificatori
   Prima fare PCA, poi allenare modelli:
   - SVM
   - Random Forest
   - KNN
   - MLP
   - Regressore Lineare

7. Inferenza e Valutazione:
   - Prendi la nuova immagine -> Resize a 224x224.
   - Passala nella CNN (Backbone) -> Ottieni il vettore di feature.
   - Applica la stessa trasformazione PCA usata nel training.
   - Passa il vettore ridotto al classificatore ML
   - Ottieni la classe finale.
8. Metriche:
   - Accuracy
   - F1-score
   - Inference time
   - model size
   - Robustezza
  
9. Creazione dataset esterno:
   - Acquisizone foto
   - Etichettatura tramite ROBOFLOW
   - Passare dati al modello e fare inferenza.
  


