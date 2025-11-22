import numpy as np
import json

class Codebook:
    def __init__(self, vector_size):
        self.vector_size = vector_size  # e.g., 16 for 4x4 blocks
        self.codewords = []             # list of numpy arrays

    def initialize_random(self, samples, k):
        indices = np.random.choice(len(samples), k, replace=False)
        self.codewords = [samples[i].copy() for i in indices]

    def nearest_codeword(self, vector):
        distances = [np.linalg.norm(vector - cw) for cw in self.codewords]
        return int(np.argmin(distances))

    def lbg_training(self, samples, k, epsilon=0.01, max_iter=100, tol=1e-5):
        """Train codebook using LBG algorithm (iterative splitting + k-means style refinement)."""
        samples = np.asarray(samples)
        # Start with global mean
        mean_vector = np.mean(samples, axis=0)
        self.codewords = [mean_vector]

        # Keep splitting until we have k codewords
        while len(self.codewords) < k:
            # split
            new_codewords = []
            for cw in self.codewords:
                new_codewords.append(cw * (1 + epsilon))
                new_codewords.append(cw * (1 - epsilon))
            self.codewords = new_codewords

            # refine with k-means style iterations
            for _ in range(max_iter):
                clusters = [[] for _ in range(len(self.codewords))]
                for v in samples:
                    idx = self.nearest_codeword(v)
                    clusters[idx].append(v)

                new_codewords = []
                for i, cluster in enumerate(clusters):
                    if len(cluster) == 0:
                        # reinitialize empty cluster with a random sample
                        new_codewords.append(self.codewords[i].copy())
                    else:
                        new_codewords.append(np.mean(cluster, axis=0))

                diff = np.sum([np.linalg.norm(self.codewords[i] - new_codewords[i]) for i in range(len(self.codewords))])
                self.codewords = new_codewords
                if diff < tol:
                    break

        # If we've created more than k (due rounding), trim
        if len(self.codewords) > k:
            self.codewords = self.codewords[:k]

    def encode(self, samples):
        return np.array([self.nearest_codeword(v) for v in samples], dtype=np.int32)

    def decode(self, indices):
        return np.array([self.codewords[i] for i in indices])

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
