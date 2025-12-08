import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
from utils.dataset import GTSRBDataset
from tqdm import tqdm
# Configurazione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


if __name__ == "__main__":
    
    # 1. Percorsi
    current_dir = os.path.dirname(os.path.abspath(__file__)) # cartella corrente
    DATA_ROOT = os.path.join(current_dir, 'data') # cartella data
    FEATURES_DIR = os.path.join(DATA_ROOT, 'features') # cartella features output
    RAW_DATA_DIR = os.path.join(current_dir, 'data/german-traffic-sign')
    CSV_PATH = os.path.join(RAW_DATA_DIR, 'Test.csv') # cambiare in Test.csv per test set
    
    os.makedirs(FEATURES_DIR, exist_ok=True)
    print(f"Output folder pronta: {FEATURES_DIR}")
    
    # 2. Trasformazioni
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Dataset e DataLoader
    try:
        dataset = GTSRBDataset(root_dir=RAW_DATA_DIR, csv_file=CSV_PATH, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    except FileNotFoundError as e:
        print(f"ERRORE: {e}")
        exit()
        
    # 4. Modello
    print("Caricamento ResNet18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    features_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

    print(f"Inizio estrazione features da {len(dataset)} immagini...")

    # 5. Loop
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, unit="batch", desc="Estrazione features"):
            images = images.to(device)
            #forward pass
            outputs = features_extractor(images)
            outputs = outputs.flatten(1) # Da [B, 512, 1, 1] a [B, 512]
            
            all_features.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    print("Concatenazione array...")
    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    print("-" * 30)
    print("Estrazione completata!")
    print(f"Shape Features: {features_array.shape}")
    print(f"Shape Labels:   {labels_array.shape}")
    
    # Salva il file nella cartella data/features
    output_filename = "features_test.npz" # cambiare in features_test.npz per test set
    save_path = os.path.join(FEATURES_DIR, output_filename)
    np.savez_compressed(save_path, features=features_array, labels=labels_array)
    print(f"Dati salvati in data/features/{output_filename}")