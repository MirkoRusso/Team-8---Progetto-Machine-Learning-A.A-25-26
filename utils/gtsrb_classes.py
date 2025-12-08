# utils/gtsrb_classes.py

GTSRB_CLASSES = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons'
}

# --- BLOCCO DI TEST / ESEMPIO ---
# Questo codice viene eseguito SOLO se lanci direttamente questo file.
# Se fai "from utils.gtsrb_classes import GTSRB_CLASSES", questo pezzo viene ignorato.
if __name__ == "__main__":
    import numpy as np
    import os
    
    print("--- Test e Esempio di Utilizzo ---")

    # 1. Calcolo dinamico del percorso per trovare il file npz
    # Siamo in /utils, dobbiamo risalire in /progetto-machine-learning
    script_dir = os.path.dirname(os.path.abspath(__file__)) # cartella utils
    project_root = os.path.dirname(script_dir)              # cartella progetto
    file_path = os.path.join(project_root, 'data', 'features', 'features_train.npz')

    if os.path.exists(file_path):
        print(f"Caricamento dati da: {file_path}")
        
        # 2. Caricamento
        data = np.load(file_path)
        X_train = data['features'] 
        classes_ids = data['labels'] 

        print(f"Shape Features: {X_train.shape}") 
        
        # 3. Conversione <------ Esempio di utilizzo
        class_names = [GTSRB_CLASSES[i] for i in classes_ids]

        print(f"\n--- Conversione Totale ---")
        print(f"Primi 5 ID: {classes_ids[:5]}")
        print(f"Primi 5 Nomi: {class_names[:5]}")
        
    else:
        print(f"ATTENZIONE: File non trovato in {file_path}")
        