import json
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import os
import math
import struct

script_dir = os.path.dirname(os.path.abspath(__file__))  # script working directory

class Codebook:
    def __init__(self, path, block_h, block_w):
        self.path = path
        self.block_h = block_h
        self.block_w = block_w

        img = Image.open(self.path).convert("RGB") # opens image as a RGB array
        self.img_arr = np.array(img)
        self.orig_h, self.orig_w, self.channels = self.img_arr.shape


        # pads the image so that its dimensions are multiples of block size
        pad_h = (self.block_h - (self.orig_h % self.block_h)) % self.block_h
        pad_w = (self.block_w - (self.orig_w % self.block_w)) % self.block_w


        # pads the image using edge pixels to avoid adding new colors to the image
        self.img_padded = np.pad(
            self.img_arr,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
            #constant_values=0
        )


        self.padded_h, self.padded_w, _ = self.img_padded.shape
        self.blocks = self.image_to_blocks()
        self.codebook = None
        self.n_rows = self.padded_h // self.block_h
        self.n_cols = self.padded_w // self.block_w

        # Output files
        self.base_name = os.path.splitext(os.path.basename(self.path))[0]
        self.codebook_json = os.path.join(script_dir, f"{self.base_name}_codebook.json")
        self.codebook_txt = os.path.join(script_dir, f"{self.base_name}_codebook.txt")
        self.labels_json = os.path.join(script_dir, f"{self.base_name}_labels.json")
        self.labels_bin = os.path.join(script_dir, f"{self.base_name}_labels.bin")
        self.reconstructed_path = os.path.join(script_dir, f"{self.base_name}_reconstructed.png")

    # creates blocks from the padded image
    def image_to_blocks(self):
        h, w, c = self.img_padded.shape
        n_rows = h // self.block_h
        n_cols = w // self.block_w
        blocks = self.img_padded.reshape(n_rows, self.block_h, n_cols, self.block_w, c)
        blocks = blocks.swapaxes(1, 2)
        return blocks.reshape(-1, self.block_h * self.block_w * c)

    
    def generate_codebook(self, k, epsilon=0.01, threshold=0.001, max_iterations=100):
        if k > len(self.blocks):
            raise ValueError(
                f"Invalid quantization level k={k}: cannot exceed the total number of image blocks ({len(self.blocks)})."
            )

        print(f"\n=== Starting LBG for k={k} ===")
        centroid = np.mean(self.blocks, axis=0) # gets the mean of all blocks as the initial centroid
        self.codebook = np.array([centroid]) # initializes the codebook with the centroid

        while len(self.codebook) < k: # while the codebook hasn't reached the desired level of quantization
            code_plus = self.codebook * (1 + epsilon) # right branch (adds a small value for percision)
            code_minus = self.codebook * (1 - epsilon) # left branch (subtracts a small value for percision)
            self.codebook = np.vstack((code_plus, code_minus)) # adds the new branches to the codebook underneath the old ones

            prev_distortion = float('inf') # association level is first set to infinity
            for i in range(max_iterations):
                distances = cdist(self.blocks, self.codebook, metric='cityblock') # calculates the Manhattan distance between each block and each codevector
                labels = np.argmin(distances, axis=1)
                new_codebook = np.zeros_like(self.codebook)

                for idx in range(len(self.codebook)): # assigns each block to the nearest codevector
                    members = self.blocks[labels == idx]
                    if len(members) > 0:
                        new_codebook[idx] = np.mean(members, axis=0)
                    else:
                        new_codebook[idx] = self.codebook[idx]

                self.codebook = new_codebook
                min_distances = distances[np.arange(len(distances)), labels] 
                distortion = np.mean(min_distances)

                if prev_distortion != float('inf'): # checks for no more movement in association level 
                    change = abs(prev_distortion - distortion) / prev_distortion
                    if change < threshold:
                        print(f"Converged at iter {i}, distortion={distortion:.3f}")
                        break

                prev_distortion = distortion

        # Save codebook as JSON
        final = self.codebook.reshape(-1, self.block_h, self.block_w, self.channels).tolist()
        with open(self.codebook_json, "w") as f:
            json.dump(final, f, indent=4)
        print(f"✓ Codebook saved to JSON: {self.codebook_json}")

        # Save codebook as TXT table
        with open(self.codebook_txt, "w") as f:
            f.write(f"{'Level':<6}{'Min':>10}{'Max':>10}{'Dequantized':>30}\n")
            f.write("-"*60 + "\n")
            for idx, vec in enumerate(self.codebook):
                min_val = vec.min()
                max_val = vec.max()
                dequant_val = np.round(vec.mean(), 2)
                f.write(f"{idx:<6}{min_val:>10.2f}{max_val:>10.2f}{dequant_val:>30.2f}\n")
        print(f"✓ Codebook saved as formatted TXT: {self.codebook_txt}")

        return final

    # assigns each block to the nearest codevector and saves the labels
    def compress(self):
        if self.codebook is None:
            raise ValueError("No codebook yet.")

        distances = cdist(self.blocks, self.codebook, metric="cityblock")
        labels = np.argmin(distances, axis=1)
        labels_grid = labels.reshape(self.n_rows, self.n_cols)

        # Save labels as JSON
        with open(self.labels_json, "w") as f:
            json.dump(labels_grid.tolist(), f)
        print(f"✓ Labels saved as JSON: {self.labels_json}")

        # Save labels as binary
        bits_needed = math.ceil(math.log2(len(self.codebook)))
        binary_data = 0
        bit_count = 0
        with open(self.labels_bin, "wb") as f:
            for lbl in labels.flatten():
                binary_data = (binary_data << bits_needed) | lbl
                bit_count += bits_needed
                while bit_count >= 8:
                    byte = (binary_data >> (bit_count - 8)) & 0xFF
                    f.write(struct.pack("B", byte))
                    bit_count -= 8
            if bit_count > 0:
                byte = (binary_data << (8 - bit_count)) & 0xFF
                f.write(struct.pack("B", byte))
        print(f"✓ Labels saved as binary: {self.labels_bin}")

        return labels_grid


    def decompress(labels_path, codebook_path, output_path):
        labels = np.array(json.load(open(labels_path)))
        codebook = np.array(json.load(open(codebook_path)))

        n_rows, n_cols = labels.shape
        block_h, block_w, channels = codebook.shape[1], codebook.shape[2], codebook.shape[3]

        img_rows = []
        for r in range(n_rows):
            block_row = [codebook[labels[r][c]] for c in range(n_cols)]
            for i in range(block_h):
                row = []
                for blk in block_row:
                    row.extend(blk[i])
                img_rows.append(row)

        arr = np.array(img_rows, dtype=np.uint8).reshape(n_rows*block_h, n_cols*block_w, channels)
        Image.fromarray(arr, "RGB").save(output_path)
        print(f"✓ Decompression done. Saved as {output_path}")
        return arr

def validate_image_path(path, allowed_exts=None):
    if allowed_exts is None:
        allowed_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    if not os.path.isabs(path):
        path = os.path.join(script_dir, path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in allowed_exts:
        raise ValueError(f"Invalid file extension '{ext}'. Allowed: {', '.join(allowed_exts)}")

    return path


if __name__ == "__main__":
    print("=+= Vector Quantization System =+=")

    while True:
        print("\nWhat would you like to do?:")
        print("1) Compress Image")
        print("2) Decompress Image")
        print("3) Exit")

        choice = input("Please choose from(1/2/3): ")

        if choice == "1":
            path = input("Enter image path: ")
            try:
                path = validate_image_path(path)
            except Exception as e:
                print("Error:", e)
                continue

            try:
                bh = int(input("Block height: "))
                bw = int(input("Block width: "))

                if bh <= 0 or bw <= 0:
                    raise ValueError("Block height and width must be positive integers.")

                # Load image first to validate against its size
                cb_temp = Codebook(path, 1, 1)  # temporary object just to get image shape
                img_h, img_w = cb_temp.orig_h, cb_temp.orig_w

                if bh > img_h or bw > img_w:
                    raise ValueError(
                        f"Block size {bh}×{bw} exceeds image size {img_h}×{img_w}."
                    )

                # Actual initialization with validated block size
                cb = Codebook(path, bh, bw)

                k = int(input("Levels of desired Quantization (size of codebook): "))

                cb.generate_codebook(k)
                cb.compress()

            except ValueError as e:
                print("Invalid input:", e)
                continue

        elif choice == "2":
            path = input("Enter original image path for output naming: ")
            try:
                path = validate_image_path(path)
            except Exception as e:
                    print("Error:", e)
                    continue
            base_name = os.path.splitext(os.path.basename(path))[0]
            labels_path = os.path.join(script_dir, f"{base_name}_labels.json")
            codebook_path = os.path.join(script_dir, f"{base_name}_codebook.json")
            reconstructed_path = os.path.join(script_dir, f"{base_name}_reconstructed.png")

            Codebook.decompress(labels_path, codebook_path, reconstructed_path)

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice.")
