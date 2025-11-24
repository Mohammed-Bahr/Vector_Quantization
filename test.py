import random
import json
from PIL import Image
import numpy as np


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


# مثال على الاستخدام
if __name__ == "__main__":
    cb = CodeBook("leaf.jpg", 2, 2)
    codebook = cb.generate_codebook(k=8, epsilon=0.01, threshold=0.001)
    
    print(f"\nNumber of codewords: {len(codebook)}")
    print(f"First codeword:\n{codebook[0]}")