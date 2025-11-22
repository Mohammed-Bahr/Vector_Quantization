import numpy as np
import json

class Codebook:
    """
    Codebook class for Vector Quantization.

    Added functionalities:
    1. image_to_2x2_vectors(image_array):
         - Takes a grayscale image as a 2D numpy array
         - Pads it if needed
         - Splits it into 2x2 blocks
         - Returns: (list of flattened 2x2 vectors, padded image shape)

    2. train_codebook_from_vectors(vectors, k):
         - Takes the generated vectors
         - Runs LBG training
         - Returns trained codebook (list of codewords)
    """

    # def __init__(self, vector_size):
    #     self.vector_size = vector_size  # e.g., 4 for 2x2 blocks
    #     self.codewords = []

    # -------------------------------------------------------------
    # 1) Convert image → 2x2 vectors
    # -------------------------------------------------------------
    def image_to_2x2_vectors(self, img_array):
        h, w = img_array.shape

        # Padding to make the image divisible by 2x2 blocks
        pad_h = (2 - (h % 2)) % 2
        pad_w = (2 - (w % 2)) % 2

        padded = np.pad(img_array,
                        ((0, pad_h), (0, pad_w)),
                        mode='constant', constant_values=0)

        blocks = []
        for i in range(0, padded.shape[0], 2):
            for j in range(0, padded.shape[1], 2):
                block = padded[i:i+2, j:j+2]
                blocks.append(block.flatten().astype(np.float32))

        return np.array(blocks), padded.shape

    # # -------------------------------------------------------------
    # # 2) Train codebook from vectors
    # # -------------------------------------------------------------
    # def train_codebook_from_vectors(self, vectors, k):
    #     self.lbg_training(vectors, k)
    #     return self.codewords

    # # -------------------------------------------------------------
    # # RANDOM INITIALIZATION
    # # -------------------------------------------------------------
    # def initialize_random(self, samples, k):
    #     indices = np.random.choice(len(samples), k, replace=False)
    #     self.codewords = [samples[i].copy() for i in indices]

    # # -------------------------------------------------------------
    # # NEAREST CODEWORD
    # # -------------------------------------------------------------
    # def nearest_codeword(self, vector):
    #     distances = [np.linalg.norm(vector - cw) for cw in self.codewords]
    #     return int(np.argmin(distances))

    # # -------------------------------------------------------------
    # # LBG TRAINING
    # # -------------------------------------------------------------
    # def lbg_training(self, samples, k, epsilon=0.01, max_iter=100, tol=1e-5):
    #     samples = np.asarray(samples)

    #     # Start with global mean
    #     mean_vector = np.mean(samples, axis=0)
    #     self.codewords = [mean_vector]

    #     # Keep splitting until reaching k
    #     while len(self.codewords) < k:
    #         new_codewords = []
    #         for cw in self.codewords:
    #             new_codewords.append(cw * (1 + epsilon))
    #             new_codewords.append(cw * (1 - epsilon))
    #         self.codewords = new_codewords

    #         # Refinement
    #         for _ in range(max_iter):
    #             clusters = [[] for _ in range(len(self.codewords))]

    #             for v in samples:
    #                 idx = self.nearest_codeword(v)
    #                 clusters[idx].append(v)

    #             new_codewords = []
    #             for i, cluster in enumerate(clusters):
    #                 if len(cluster) == 0:
    #                     new_codewords.append(self.codewords[i].copy())
    #                 else:
    #                     new_codewords.append(np.mean(cluster, axis=0))

    #             diff = np.sum([
    #                 np.linalg.norm(self.codewords[i] - new_codewords[i])
    #                 for i in range(len(self.codewords))
    #             ])

    #             self.codewords = new_codewords
    #             if diff < tol:
    #                 break

    #     # Trim if exceeded k
    #     if len(self.codewords) > k:
    #         self.codewords = self.codewords[:k]

    # # -------------------------------------------------------------
    # # ENCODE / DECODE
    # # -------------------------------------------------------------
    # def encode(self, samples):
    #     return np.array([self.nearest_codeword(v) for v in samples], dtype=np.int32)

    # def decode(self, indices):
    #     return np.array([self.codewords[i] for i in indices])

    # -------------------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------------------
    def save(self, filepath):
        data = {
            "vector_size": self.vector_size,
            "codewords": [cw.tolist() for cw in self.codewords]
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        cb = Codebook(data["vector_size"])
        cb.codewords = [np.array(cw) for cw in data["codewords"]]
        return cb