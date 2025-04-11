import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Specifica il percorso del file che vuoi visualizzare
file_path = 'CVPR2024-FACT/attn/a2f_attn_0.npy'  # cambia il nome se necessario

# Controlla che il file esista
if not os.path.exists(file_path):
    print(f"Errore: il file '{file_path}' non esiste.")
else:
    # Carica la mappa di attenzione
    attn = np.load(file_path)

    # Verifica che sia una matrice 2D
    if len(attn.shape) != 2:
        print(f"Errore: il file contiene un array con {attn.ndim} dimensioni invece di 2.")
    else:
        print(f"Caricata mappa con shape {attn.shape} (T x M)")

        # Visualizzazione con heatmap
        plt.figure(figsize=(10, 4))
        sns.heatmap(attn.T, cmap="viridis", cbar=True)
        plt.xlabel("Frame Index")
        plt.ylabel("Token Index")
        plt.title("Frame-to-Token Attention Map (Λf)")
        plt.tight_layout()
        plt.show()