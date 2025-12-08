import numpy as np
import json
from scipy.spatial.distance import cdist

def generate_codebook_uniform(self, k):
    """
    Build a uniform scalar quantizer codebook with k levels and save it.
    Assumes:
      - self.blocks: numpy array shape (N, D) where D = block_h * block_w * channels
      - self.block_h, self.block_w, self.channels exist for reshaping when saving
      - self.codebook_json, self.codebook_txt are paths for output
    Returns:
      - final: list representation of the codebook reshaped to (k, block_h, block_w, channels)
    """
    if k <= 0:
        raise ValueError("k must be >= 1")

    # use the actual data range (robust) — you can switch to fixed 0..255 if you prefer
    global_min = float(np.min(self.blocks))
    global_max = float(np.max(self.blocks))

    if global_max == global_min:
        # degenerate case: all values equal -> single level
        levels = np.array([global_min])
    else:
        Delta = (global_max - global_min) / k
        # mid-point levels
        levels = global_min + (np.arange(k) + 0.5) * Delta

    # Build codebook: for scalar quantizer we represent each level as a vector
    # repeated across the block vector dimension so that codebook has shape (k, D)
    D = self.blocks.shape[1]
    codebook = np.repeat(levels.reshape(-1, 1), D, axis=1)  # shape (k, D)

    # Optionally: if you want to compute and store quantized blocks (labels), you can:
    # distances = cdist(self.blocks, codebook, metric='cityblock')
    # labels = np.argmin(distances, axis=1)
    # quantized_blocks = codebook[labels]

    self.codebook = codebook  # keep for later use

    # Save codebook as JSON (reshaped to block shape)
    final = self.codebook.reshape(-1, self.block_h, self.block_w, self.channels).tolist()
    with open(self.codebook_json, "w") as f:
        json.dump(final, f, indent=4)
    print(f"✓ Uniform scalar codebook saved to JSON: {self.codebook_json}")

    # Save codebook as TXT table (one row per level)
    with open(self.codebook_txt, "w") as f:
        f.write(f"{'Level':<6}{'LevelValue':>12}{'MinInRange':>14}{'MaxInRange':>14}\n")
        f.write("-" * 52 + "\n")
        if global_max == global_min:
            # only one level
            f.write(f"{0:<6}{global_min:>12.2f}{global_min:>14.2f}{global_max:>14.2f}\n")
        else:
            # compute interval boundaries for clarity
            boundaries = [global_min + i * (global_max - global_min) / k for i in range(k + 1)]
            for idx, lvl in enumerate(levels):
                bmin = boundaries[idx]
                bmax = boundaries[idx + 1]
                f.write(f"{idx:<6}{lvl:>12.2f}{bmin:>14.2f}{bmax:>14.2f}\n")

    print(f"✓ Uniform scalar codebook saved as formatted TXT: {self.codebook_txt}")

    return final
