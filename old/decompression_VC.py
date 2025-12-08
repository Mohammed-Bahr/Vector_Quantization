import numpy as np
from PIL import Image

# ============================
# CODEBOOK EXAMPLE
# ============================

codebook = {
    "00": [[2, 3],
           [4, 5]],
    
    "01": [[8, 9],
           [7, 7]],
    
    "10": [[4, 10],
           [11, 11]],
    
    "11": [[16, 15],
           [19, 18]]
}

# ============================
# COMPRESSED IMAGE EXAMPLE
# ============================

compressed = [
    ["00", "01", "10"],
    ["10", "11", "01"],
    ["00", "11", "00"]
]


# ============================
# DECODING FUNCTION
# ============================

def decode_image(compressed_matrix, codebook):
    reconstructed = []
    for row in compressed_matrix:
        row_blocks = []
        for code in row:
            row_blocks.append(codebook[code])
        reconstructed.append(row_blocks)
    
    # Reconstruct full image from blocks
    blocks = reconstructed
    block_size = len(blocks[0][0])
    rows = []
    
    for block_row in blocks:
        for i in range(block_size):
            row = []
            for block in block_row:
                row.extend(block[i])
            rows.append(row)
    return rows

# ============================
# SAVE IMAGE
# ============================

def save_image(matrix, filename="reconstructed.png"):
    arr = np.array(matrix, dtype=np.uint8)
    img = Image.fromarray(arr, 'L')
    img.save(filename)
    print(f"Image saved as {filename}")

# ============================
# RUNNING THE DECODING
# ============================

full_matrix_decoded = decode_image(compressed, codebook)
save_image(full_matrix_decoded)

# Print the decoded matrix
for row in full_matrix_decoded:
    print(row)