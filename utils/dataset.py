from torchvision import transforms, datasets
from torch.utils.data import DataLoader 
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]
    )
])

class GTSRBDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_name = self.csv_data.iloc[idx]['Path'] 
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.csv_data.iloc[idx]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # --- CALCOLO DINAMICO DEI PERCORSI ---   
    # 1. Trova dov'è questo file 
    current_script_path = os.path.abspath(__file__)
    
    # 2. Risale di due livelli per trovare la root del progetto (progetto-machine-learning)
    #    dirname(script) -> utils
    #    dirname(utils)  -> progetto-machine-learning
    PROJECT_ROOT = os.path.dirname(os.path.dirname(current_script_path))
    
    # 3. Costruisce il percorso verso i dati partendo dalla root
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'german-traffic-sign')
    
    # 4. Definisce i percorsi dei CSV
    TRAIN_CSV = os.path.join(DATA_ROOT, 'Train.csv')
    TEST_CSV = os.path.join(DATA_ROOT, 'Test.csv')

    print(f"Project Root rilevata: {PROJECT_ROOT}")
    print(f"Data Root impostata: {DATA_ROOT}")

    # --- TEST CARICAMENTO ---
    try:
        
        train_ds = GTSRBDataset(root_dir=DATA_ROOT, csv_file=TRAIN_CSV, transform=data_transforms)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        
        
        images, labels = next(iter(train_loader))
        print(f"\nSUCCESS! Batch caricato correttamente.")
        print(f"Shape Immagini: {images.shape}") # [4, 3, 224, 224]
        print(f"Shape Labels: {labels.shape}")   # [4]

    except Exception as e:
        print(f"\nERRORE: Qualcosa non va nei percorsi.\n{e}")