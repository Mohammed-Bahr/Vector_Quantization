# import numpy as np
# import json

# class Codebook:
#     def __init__(self, vector_size):
#         self.vector_size = vector_size  # e.g., 16 for 4x4 blocks
#         self.codewords = []             # list of numpy arrays

#     def initialize_random(self, samples, k):
#         indices = np.random.choice(len(samples), k, replace=False)
#         self.codewords = [samples[i].copy() for i in indices]

#     def nearest_codeword(self, vector):
#         distances = [np.linalg.norm(vector - cw) for cw in self.codewords]
#         return int(np.argmin(distances))

#     def lbg_training(self, samples, k, epsilon=0.01, max_iter=100, tol=1e-5):
#         """Train codebook using LBG algorithm (iterative splitting + k-means style refinement)."""
#         samples = np.asarray(samples)
#         # Start with global mean
#         mean_vector = np.mean(samples, axis=0)
#         self.codewords = [mean_vector]

#         # Keep splitting until we have k codewords
#         while len(self.codewords) < k:
#             # split
#             new_codewords = []
#             for cw in self.codewords:
#                 new_codewords.append(cw * (1 + epsilon))
#                 new_codewords.append(cw * (1 - epsilon))
#             self.codewords = new_codewords

#             # refine with k-means style iterations
#             for _ in range(max_iter):
#                 clusters = [[] for _ in range(len(self.codewords))]
#                 for v in samples:
#                     idx = self.nearest_codeword(v)
#                     clusters[idx].append(v)

#                 new_codewords = []
#                 for i, cluster in enumerate(clusters):
#                     if len(cluster) == 0:
#                         # reinitialize empty cluster with a random sample
#                         new_codewords.append(self.codewords[i].copy())
#                     else:
#                         new_codewords.append(np.mean(cluster, axis=0))

#                 diff = np.sum([np.linalg.norm(self.codewords[i] - new_codewords[i]) for i in range(len(self.codewords))])
#                 self.codewords = new_codewords
#                 if diff < tol:
#                     break

#         # If we've created more than k (due rounding), trim
#         if len(self.codewords) > k:
#             self.codewords = self.codewords[:k]

#     def encode(self, samples):
#         return np.array([self.nearest_codeword(v) for v in samples], dtype=np.int32)

#     def decode(self, indices):
#         return np.array([self.codewords[i] for i in indices])

#     def save(self, filepath):
#         data = {
#             "vector_size": self.vector_size,
#             "codewords": [cw.tolist() for cw in self.codewords]
#         }
#         with open(filepath, "w") as f:
#             json.dump(data, f)

#     @staticmethod
#     def load(filepath):
#         with open(filepath, "r") as f:
#             data = json.load(f)
#         cb = Codebook(data["vector_size"])
import numpy as np
import json

class Codebook:
    def __init__(self, vector_size):
        self.vector_size = vector_size
        self.codewords = []

    def nearest_codeword(self, vector):
        """Find index of closest codeword to given vector."""
        # This is kept for single vector usage, but internal methods should use vectorized versions
        if not self.codewords:
            return -1
        codewords = np.array(self.codewords)
        dists = np.linalg.norm(codewords - vector, axis=1)
        return np.argmin(dists)

    def lbg_training(self, samples, k, epsilon=0.01, max_iter=100, tol=1e-5):
        """Train codebook using LBG algorithm with vectorized operations."""
        samples = np.asarray(samples)
        
        # Start with mean of all samples
        mean_vector = np.mean(samples, axis=0)
        self.codewords = [mean_vector]

        # Split codewords until we reach k
        while len(self.codewords) < k:
            # Split each codeword into two
            new_codewords = []
            for codeword in self.codewords:
                new_codewords.append(codeword * (1 + epsilon))
                new_codewords.append(codeword * (1 - epsilon))
            
            current_codewords = np.array(new_codewords)
            
            # Refine codewords using k-means
            for iteration in range(max_iter):
                # Vectorized assignment
                # samples: (N, D), codewords: (K, D)
                # dists: (N, K)
                
                # To avoid memory issues with large N*K, we can process in chunks if needed.
                # For now, assuming reasonable size or letting numpy handle it.
                # Using broadcasting: (N, 1, D) - (1, K, D)
                
                # Optimization: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
                # This is faster and uses less memory than (N,1,D)-(1,K,D)
                
                N = samples.shape[0]
                K = len(current_codewords)
                
                # Simple broadcasting for correctness and readability first
                # If N is large, this might OOM. Let's use a loop over chunks if N is very large.
                # But for typical assignment usage, let's try the direct broadcast if N*K*D < 1e8 elements.
                
                # Let's use a safer chunked approach for assignment
                assignments = np.zeros(N, dtype=int)
                chunk_size = 1000
                
                for i in range(0, N, chunk_size):
                    end = min(i + chunk_size, N)
                    batch = samples[i:end]
                    # (B, 1, D) - (1, K, D)
                    dists = np.linalg.norm(batch[:, np.newaxis, :] - current_codewords[np.newaxis, :, :], axis=2)
                    assignments[i:end] = np.argmin(dists, axis=1)

                # Update centroids
                new_centers = []
                total_change = 0
                
                for i in range(K):
                    assigned_mask = (assignments == i)
                    if np.any(assigned_mask):
                        new_center = np.mean(samples[assigned_mask], axis=0)
                        new_centers.append(new_center)
                        total_change += np.linalg.norm(current_codewords[i] - new_center)
                    else:
                        # Keep old codeword if no samples assigned
                        new_centers.append(current_codewords[i])
                
                new_centers = np.array(new_centers)
                
                if total_change < tol:
                    current_codewords = new_centers
                    break
                
                current_codewords = new_centers

            self.codewords = [c for c in current_codewords]

        # Keep only k codewords
        if len(self.codewords) > k:
            self.codewords = self.codewords[:k]

    def encode(self, samples):
        """Encode samples to codeword indices using vectorized search."""
        samples = np.asarray(samples)
        if not self.codewords:
            return np.zeros(len(samples), dtype=int)
            
        codewords = np.array(self.codewords)
        N = samples.shape[0]
        indices = np.zeros(N, dtype=int)
        
        chunk_size = 1000
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            batch = samples[i:end]
            dists = np.linalg.norm(batch[:, np.newaxis, :] - codewords[np.newaxis, :, :], axis=2)
            indices[i:end] = np.argmin(dists, axis=1)
            
        return indices

    def decode(self, indices):
        """Decode indices back to vectors."""
        codewords = np.array(self.codewords)
        return codewords[indices]

    def save(self, filepath):
        """Save codebook to JSON file."""
        data = {
            "vector_size": self.vector_size,
            "codewords": [cw.tolist() for cw in self.codewords]
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(filepath):
        """Load codebook from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        cb = Codebook(data["vector_size"])
        cb.codewords = [np.array(cw) for cw in data["codewords"]]
        return cb