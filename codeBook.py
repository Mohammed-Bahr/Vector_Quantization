import random
import json
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist


class CodeBook:
    
    def __init__(self, path, block_h, block_w):
        self.path = path
        self.block_h = block_h
        self.block_w = block_w
        self.codebook = []
        self.blocks = self.image_to_blocks()  
        self.centroid = self.compute_average_block()  # متوسط كل البلوكات

     
    def image_to_blocks(self):
        image_path = self.path
        block_width = self.block_w
        block_height = self.block_h

        img = Image.open(image_path).convert("L")
        img = np.array(img)

        img_height, img_width = img.shape

        blocks = []

        for y in range(0, img_height, block_height):
            for x in range(0, img_width, block_width):

                block = img[y:y+block_height, x:x+block_width]

                if block.shape != (block_height, block_width):
                    padded = np.zeros((block_height, block_width), dtype=img.dtype)
                    padded[:block.shape[0], :block.shape[1]] = block
                    block = padded

                blocks.append(block.tolist())

        return blocks


    def compute_average_block(self):
        """ حساب متوسط كل القيم في كل بلوك """
        total_blocks = len(self.blocks)

        h = self.block_h
        w = self.block_w

        avg = [[0 for _ in range(w)] for _ in range(h)]

        for block in self.blocks:
            for i in range(h):
                for j in range(w):
                    avg[i][j] += block[i][j]

        # القسمة على عدد البلوكات
        for i in range(h):
            for j in range(w):
                avg[i][j] /= total_blocks

        return avg


    def block_difference(self, block1, block2):
        """ الفرق المطلق بين قيم كل مكان في البلوك """
        total = 0
        h = len(block1)
        w = len(block1[0])

        for i in range(h):
            for j in range(w):
                total += abs(block1[i][j] - block2[i][j])

        return total


    # def generate_codebook(self, k, json_path="codebook.json"):
    #     """
    #     توليد كودبوك من k بلوكات عشوائية
    #     ثم حفظه في ملف JSON
    #     """
    #     if k > len(self.blocks):
    #         raise ValueError("k is larger than number of blocks!")

    #     self.codebook = random.sample(self.blocks, k)

    #     # كتابة JSON
    #     with open(json_path, "w") as f:
    #         json.dump(self.codebook, f, indent=4)

    #     return self.codebook

    

    def generate_codebook(self, k, json_path="codebook.json", epsilon=0.01, threshold=0.001, max_iterations=100):
        """
        توليد كودبوك باستخدام خوارزمية LBG الكاملة
        
        Parameters:
        -----------
        k : int
            حجم الكودبوك المطلوب (يفضل أن يكون قوة 2)
        json_path : str
            مسار حفظ ملف JSON
        epsilon : float
            قيمة الاضطراب لتقسيم الكودووردز
        threshold : float
            حد التوقف عند التقارب
        max_iterations : int
            أقصى عدد تكرارات لكل مرحلة
        """
        
        if k > len(self.blocks):
            raise ValueError("k is larger than number of blocks!")
        
        # الخطوة 1: البداية بكودوورد واحد (المتوسط العام)
        self.codebook = [self.centroid]
        current_size = 1
        
        print(f"Starting LBG algorithm to generate codebook of size {k}...")
        
        # الخطوة 2: التكرار حتى الوصول للحجم المطلوب
        while current_size < k:
            # تقسيم كل كودوورد إلى اثنين
            new_codebook = []
            for codeword in self.codebook:
                # إضافة اضطراب موجب
                codeword_plus = []
                for i in range(len(codeword)):
                    row = []
                    for j in range(len(codeword[0])):
                        row.append(codeword[i][j] * (1 + epsilon))
                    codeword_plus.append(row)
                
                # إضافة اضطراب سالب
                codeword_minus = []
                for i in range(len(codeword)):
                    row = []
                    for j in range(len(codeword[0])):
                        row.append(codeword[i][j] * (1 - epsilon))
                    codeword_minus.append(row)
                
                new_codebook.append(codeword_plus)
                new_codebook.append(codeword_minus)
            
            self.codebook = new_codebook
            current_size = len(self.codebook)
            
            print(f"\nCodebook size after splitting: {current_size}")
            
            # الخطوة 3: تحسين الكودبوك بالتكرار
            prev_distortion = float('inf')
            
            for iteration in range(max_iterations):
                # 3.1: تعيين كل بلوك لأقرب كودوورد
                clusters = [[] for _ in range(current_size)]
                
                for block in self.blocks:
                    min_dist = float('inf')
                    closest_idx = 0
                    
                    for idx, codeword in enumerate(self.codebook):
                        dist = self.block_difference(block, codeword)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                    
                    clusters[closest_idx].append(block)
                
                # 3.2: تحديث كل كودوورد بمتوسط الكلستر الخاص به
                updated_codebook = []
                for cluster in clusters:
                    if len(cluster) > 0:
                        # حساب المتوسط
                        h = self.block_h
                        w = self.block_w
                        avg = [[0 for _ in range(w)] for _ in range(h)]
                        
                        for block in cluster:
                            for i in range(h):
                                for j in range(w):
                                    avg[i][j] += block[i][j]
                        
                        for i in range(h):
                            for j in range(w):
                                avg[i][j] /= len(cluster)
                        
                        updated_codebook.append(avg)
                    else:
                        # إذا كان الكلستر فارغ، نبقي على الكودوورد القديم
                        cluster_idx = clusters.index(cluster)
                        updated_codebook.append(self.codebook[cluster_idx])
                
                self.codebook = updated_codebook
                
                # 3.3: حساب التشوه الكلي
                current_distortion = 0
                for idx, cluster in enumerate(clusters):
                    for block in cluster:
                        dist = self.block_difference(block, self.codebook[idx])
                        current_distortion += dist
                
                # 3.4: فحص التقارب
                if prev_distortion != float('inf'):
                    distortion_change = abs(prev_distortion - current_distortion) / prev_distortion
                    
                    if distortion_change < threshold:
                        print(f"  Converged after {iteration + 1} iterations (distortion: {current_distortion:.2f})")
                        break
                
                prev_distortion = current_distortion
                
                if iteration == max_iterations - 1:
                    print(f"  Reached max iterations ({max_iterations}), distortion: {current_distortion:.2f}")
        
        print(f"\n✓ LBG algorithm completed! Final codebook size: {len(self.codebook)}")
        
        # حفظ الكودبوك في ملف JSON
        with open(json_path, "w") as f:
            json.dump(self.codebook, f, indent=4)
        
        print(f"✓ Codebook saved to {json_path}")
        
        return self.codebook



class CodeBookOptimized:
    
    def __init__(self, path, block_h, block_w):
        self.path = path
        self.block_h = block_h
        self.block_w = block_w
        # We will store blocks as a flattened 2D array (N_blocks, Block_Size)
        self.blocks = self.image_to_blocks()  
        self.codebook = None

    def image_to_blocks(self):
        img = Image.open(self.path).convert("L")
        img_arr = np.array(img)
        h, w = img_arr.shape
        
        # Calculate padding if image dimensions are not divisible by block size
        pad_h = (self.block_h - (h % self.block_h)) % self.block_h
        pad_w = (self.block_w - (w % self.block_w)) % self.block_w
        
        # Apply padding instantly using numpy
        if pad_h > 0 or pad_w > 0:
            img_arr = np.pad(img_arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
        h, w = img_arr.shape
        
        # Reshape into blocks without loops
        # Shape becomes: (rows, block_h, cols, block_w)
        n_rows = h // self.block_h
        n_cols = w // self.block_w
        
        blocks = img_arr.reshape(n_rows, self.block_h, n_cols, self.block_w)
        
        # Swap axes to group blocks: (rows, cols, block_h, block_w)
        blocks = blocks.swapaxes(1, 2)
        
        # Flatten into (N_blocks, vector_size)
        # vector_size = block_h * block_w
        return blocks.reshape(-1, self.block_h * self.block_w)

    def generate_codebook(self, k, json_path="codebook.json", epsilon=0.01, threshold=0.001, max_iterations=100):
        if k > len(self.blocks):
            raise ValueError("k is larger than number of blocks!")

        print(f"Starting Optimized LBG for k={k}...")
        
        # Step 1: Initialize with global centroid
        centroid = np.mean(self.blocks, axis=0)
        self.codebook = np.array([centroid])
        
        current_distortion = float('inf')
        
        # Step 2: Splitting Loop
        while len(self.codebook) < k:
            # Split codewords
            codebook_plus = self.codebook * (1 + epsilon)
            codebook_minus = self.codebook * (1 - epsilon)
            self.codebook = np.concatenate([codebook_plus, codebook_minus], axis=0)
            
            # Refinement Loop (K-Means / LBG)
            prev_distortion = float('inf')
            
            for i in range(max_iterations):
                # 3.1 Calculate Distances (Vectorized)
                # cdist with 'cityblock' = Manhattan distance (same as your original Abs Diff)
                distances = cdist(self.blocks, self.codebook, metric='cityblock')
                
                # Assign blocks to nearest codeword
                labels = np.argmin(distances, axis=1)
                
                # 3.2 Calculate new centroids
                # Using numpy indexing is much faster than looping
                new_codebook = np.zeros_like(self.codebook)
                
                # Loop only through the number of clusters (K), not the number of blocks
                for idx in range(len(self.codebook)):
                    mask = labels == idx
                    if np.any(mask):
                        new_codebook[idx] = np.mean(self.blocks[mask], axis=0)
                    else:
                        # If cluster is empty, keep old codeword
                        new_codebook[idx] = self.codebook[idx]
                
                self.codebook = new_codebook
                
                # 3.3 Calculate Distortion
                # Select the distance to the chosen centroid for each block
                min_distances = distances[np.arange(len(distances)), labels]
                avg_distortion = np.mean(min_distances)
                
                # 3.4 Convergence Check
                if prev_distortion != float('inf'):
                    change = abs(prev_distortion - avg_distortion) / prev_distortion
                    if change < threshold:
                        print(f"  Converged for size {len(self.codebook)} at iter {i} (Distortion: {avg_distortion:.2f})")
                        break
                
                prev_distortion = avg_distortion
        
        print(f"✓ Codebook generated. Final size: {len(self.codebook)}")
        
        # Reshape back to 2D blocks for JSON saving to match original format
        final_codebook_list = self.codebook.reshape(-1, self.block_h, self.block_w).tolist()
        
        with open(json_path, "w") as f:
            json.dump(final_codebook_list, f, indent=4)
            
        return final_codebook_list

# Usage
if __name__ == "__main__":
    # Make sure you have a file named leaf.jpg or change the path
    try:
        cb = CodeBookOptimized("leaf.jpg", 4, 4)
        codebook = cb.generate_codebook(k=16) # Try higher K, it will still be fast
    except Exception as e:
        print(f"Error: {e}")
