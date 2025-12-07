# Team-8---Progetto-Machine-Learning-A.A-25-26


# 💻 Pipeline
### Transfer Learning:
1. Rete pre trained: ResNet / VGG16
2. Freeze layer finali
3. Data Pipeline:
   - Resize Immagini
   - Preprocessing.
4. Estrazione delle Features(lavorare in batch):
   - Feed-forward
   - Flattening
   - Salvataggio label
6. Addestramento dei seguenti classificatori
   Prima fare PCA, poi allenare modelli:
   - SVM
   - Random Forest
   - KNN
   - MLP
   - Regressore Lineare

7. Inferenza e Valutazione
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
  


