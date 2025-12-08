import kagglehub
import shutil
import os

def download_dataset():
    # 1. Scarica l'ultima versione 
    print("⏳ Inizio download da KaggleHub...")
    cached_path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    print(f"✅ Scaricato nella cache di sistema: {cached_path}")

    # 2. Calcolo dei percorsi 
    # Trova la posizione di QUESTO file 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Risale alla root del progetto (progetto-machine-learning)
    project_root = os.path.dirname(script_dir)
    
    # Costruisce il percorso verso la cartella di destinazione
    destination_folder = os.path.join(
        project_root, 
        "data", 
        "german-traffic-sign"
    )

    print(f"📂 Cartella di destinazione: {destination_folder}")

    # 3. Copia i file
    try:
        
        shutil.copytree(cached_path, destination_folder, dirs_exist_ok=True)
        print(f"Dataset copiato con successo in: {destination_folder}")
    except Exception as e:
        print(f"Errore durante la copia: {e}")

if __name__ == "__main__":
    download_dataset()