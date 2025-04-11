import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Choose which attention map to visualize ('a2f' or 'f2a')
attention_type = 'a2f'  # Change to 'f2a' for frame-to-action attention
# attention_type = 'f2a'  # Change to 'a2f' for action-to-frame attention


# File path construction
file_path = f'CVPR2024-FACT/attn/{attention_type}_attn_0.npy'

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' does not exist.")
else:
    # Load attention map
    attn = np.load(file_path)

    # Handle batch dimension (1, T, M)
    if attn.ndim == 3 and attn.shape[0] == 1:
        attn = attn[0]  # becomes (T, M)
    elif attn.ndim != 2:
        print(f"Error: Array has shape {attn.shape}, expected 2D or 3D with batch size 1.")
        exit()

    print(f"Loaded attention map with shape {attn.shape} (T x M)")

    # Plot heatmap
    plt.figure(figsize=(12, 5))
    
    # Determine titles and labels based on attention type
    if attention_type == 'a2f':
        title = "Action-to-Frame Attention Map (Λᶠ)"
        ylabel = "Token Index"
        xlabel = "Frame Index"
    else:
        title = "Frame-to-Action Attention Map (Λᵃ)"
        ylabel = "Frame Index"
        xlabel = "Token Index"

    sns.heatmap(attn.T, cmap="viridis", cbar=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()