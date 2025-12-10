import random
import json
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist


class Codebook:
    
    def __init__(self, path, block_h, block_w):
        self.path = path
        self.block_h = block_h
        self.block_w = block_w
        self.blocks = self.image_to_blocks()  
        self.codebook = None

    def image_to_blocks(self):
        img = Image.open(self.path).convert("L")
        img_arr = np.array(img)
        h, w = img_arr.shape
        
        pad_h = (self.block_h - (h % self.block_h)) % self.block_h
        pad_w = (self.block_w - (w % self.block_w)) % self.block_w
        
        if pad_h > 0 or pad_w > 0:
            img_arr = np.pad(img_arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
        h, w = img_arr.shape
        
        n_rows = h // self.block_h
        n_cols = w // self.block_w
        
        blocks = img_arr.reshape(n_rows, self.block_h, n_cols, self.block_w)
        
        blocks = blocks.swapaxes(1, 2)
        
        return blocks.reshape(-1, self.block_h * self.block_w)

    def generate_codebook(self, k, json_path="codebook.json", epsilon=0.01, threshold=0.001, max_iterations=100):
        if k > len(self.blocks):
            raise ValueError("k is larger than number of blocks!")

        print(f"Starting Optimized LBG for k={k}...")
        
        centroid = np.mean(self.blocks, axis=0)
        self.codebook = np.array([centroid])
        
        current_distortion = float('inf')
        
        while len(self.codebook) < k:
            codebook_plus = self.codebook * (1 + epsilon)
            codebook_minus = self.codebook * (1 - epsilon)
            self.codebook = np.concatenate([codebook_plus, codebook_minus], axis=0)
            
            prev_distortion = float('inf')
            
            for i in range(max_iterations):
                distances = cdist(self.blocks, self.codebook, metric='cityblock')
                
                labels = np.argmin(distances, axis=1)
                
                new_codebook = np.zeros_like(self.codebook)
                
                for idx in range(len(self.codebook)):
                    mask = labels == idx
                    if np.any(mask):
                        new_codebook[idx] = np.mean(self.blocks[mask], axis=0)
                    else:
                        new_codebook[idx] = self.codebook[idx]
                
                self.codebook = new_codebook
                
                min_distances = distances[np.arange(len(distances)), labels]
                avg_distortion = np.mean(min_distances)
                
                if prev_distortion != float('inf'):
                    change = abs(prev_distortion - avg_distortion) / prev_distortion
                    if change < threshold:
                        print(f"  Converged for size {len(self.codebook)} at iter {i} (Distortion: {avg_distortion:.2f})")
                        break
                
                prev_distortion = avg_distortion
        
        print(f"✓ Codebook generated. Final size: {len(self.codebook)}")
        
        final_codebook_list = self.codebook.reshape(-1, self.block_h, self.block_w).tolist()
        
        with open(json_path, "w") as f:
            json.dump(final_codebook_list, f, indent=4)
            
        return final_codebook_list

def compress(self, compressed_path="compressed.png", as_reconstruction=True):
        if self.codebook is None:
            raise ValueError("Codebook not generated yet. Run generate_codebook first.")

        distances = cdist(self.blocks, self.codebook, metric='cityblock')
        labels = np.argmin(distances, axis=1)

        if as_reconstruction:
            reconstructed_blocks = self.codebook[labels]
            blocks_reshaped = reconstructed_blocks.reshape(self.n_rows, self.n_cols, self.block_h, self.block_w)
            blocks_swapped = blocks_reshaped.swapaxes(1, 2)
            reconstructed_image = blocks_swapped.reshape(self.padded_h, self.padded_w)
            reconstructed_image = reconstructed_image[: self.orig_h, : self.orig_w]
            img = Image.fromarray(np.clip(reconstructed_image, 0, 255).astype(np.uint8), mode='L')
            img.save(compressed_path)
            print(f"✓ Compression complete. Saved reconstructed image to {compressed_path}")
            return labels
        else:
            max_label = np.max(labels) if np.max(labels) > 0 else 1
            map_pixels = (labels / max_label * 255).astype(np.uint8)
            map_img = map_pixels.reshape(self.n_rows, self.n_cols)
            upscaled = np.kron(map_img, np.ones((self.block_h, self.block_w), dtype=np.uint8))
            upscaled = upscaled[: self.orig_h, : self.orig_w]
            img = Image.fromarray(upscaled, mode='L')
            img.save(compressed_path)
            print(f"✓ Compression complete. Saved label-map image to {compressed_path}")
            return labels

