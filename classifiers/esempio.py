import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)


from utils.gtsrb_classes import GTSRB_CLASSES
import numpy as np

data = np.load('data/features/features_train.npz')

X_train = data['features'] # Le features estratte dalla ResNet
classes_ids = data['labels']  # Le classi (0-42)

print(X_train.shape) # Output: (39209, 512)
class_names = [GTSRB_CLASSES[i] for i in classes_ids]

print(f"\n--- Conversione Totale ---")
print(f"Primi 5 nomi nella lista: {class_names[:5]}")