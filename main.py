"""
Complete Vector Quantizer project (single-file).
Features:
- Load image (PIL)
- Convert to grayscale
- Accept dynamic inputs: image path, block size OR number of blocks
- Handle padding
- Menu: compress / decompress / exit
- LBG algorithm to train codebook
- Save codebook (JSON) and compressed indices (NPZ)
- Reconstruct and save decompressed image
- Compute compression ratio
- Save metadata including release/version timestamp

Usage: run the script (python vq_project.py) and follow prompts.

Author: generated for student project
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
from PIL import Image
from codeBook import Codebook

# ----------------------- Utility functions -----------------------

def load_image_grayscale(path):
    img = Image.open(path).convert('L')  # grayscale
    arr = np.array(img, dtype=np.uint8)
    return arr, img.mode


def save_image_from_array(arr, path):
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)


def pad_image(array, block_h, block_w):
    h, w = array.shape
    pad_h = (block_h - (h % block_h)) % block_h
    pad_w = (block_w - (w % block_w)) % block_w
    if pad_h == 0 and pad_w == 0:
        return array, (0,0)
    padded = np.pad(array, ((0,pad_h),(0,pad_w)), mode='constant', constant_values=0)
    return padded, (pad_h, pad_w)


def blocks_from_image(array, block_h, block_w):
    h, w = array.shape
    blocks = []
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            blk = array[i:i+block_h, j:j+block_w]
            blocks.append(blk.flatten().astype(np.float32))
    return np.array(blocks)


def image_from_blocks(blocks, padded_shape, block_h, block_w):
    h_p, w_p = padded_shape
    img = np.zeros((h_p, w_p), dtype=np.float32)
    idx = 0
    for i in range(0, h_p, block_h):
        for j in range(0, w_p, block_w):
            block = blocks[idx].reshape((block_h, block_w))
            img[i:i+block_h, j:j+block_w] = block
            idx += 1
    return img


def compute_sizes(original_array, block_h, block_w, codebook_size):
    # Original size in bytes (grayscale, 1 byte per pixel)
    orig_bytes = original_array.size
    # Compressed size estimate:
    # - store codebook: codebook_size * vector_size bytes (float32 -> 4 bytes each) -> we'll store as float32
    # - store indices: number_of_blocks * bits_per_index (we'll store as int32 -> 4 bytes each)
    h, w = original_array.shape
    padded, _ = pad_image(original_array, block_h, block_w)
    blocks = blocks_from_image(padded, block_h, block_w)
    num_blocks = blocks.shape[0]
    vector_size = block_h * block_w
    codebook_bytes = codebook_size * vector_size * 4  # float32
    indices_bytes = num_blocks * 4  # int32
    compressed_bytes = codebook_bytes + indices_bytes
    return orig_bytes, compressed_bytes


# ----------------------- Core interactive flow -----------------------

def compress_flow():
    path = input('Enter image path to compress: ').strip()
    if not os.path.isfile(path):
        print('File not found.'); return

    arr, _ = load_image_grayscale(path)
    print(f'Loaded image: {arr.shape[0]}x{arr.shape[1]} (grayscale)')

    mode = ''
    while mode not in ('1','2'):
        print('Choose input mode:')
        print('1) Specify block height & width (recommended)')
        print('2) Specify desired number of blocks (image will be partitioned to nearest grid)')
        mode = input('Enter 1 or 2: ').strip()

    if mode == '1':
        bh = int(input('Block height (e.g., 4): ').strip())
        bw = int(input('Block width (e.g., 4): ').strip())
    else:
        target_blocks = int(input('Desired number of blocks (approx): ').strip())
        # choose almost-square blocks
        h, w = arr.shape
        area = h * w
        block_area = max(1, area // target_blocks)
        side = int(np.sqrt(block_area))
        bh = max(1, side)
        bw = max(1, side)
        print(f'Estimated block size: {bh}x{bw}')

    padded, pad_info = pad_image(arr, bh, bw)
    print(f'Applied padding: {pad_info}, padded shape: {padded.shape}')

    samples = blocks_from_image(padded, bh, bw)
    print(f'Number of blocks (vectors): {samples.shape[0]}, vector size: {samples.shape[1]}')

    # choose codebook size
    k = int(input('Enter codebook size K (e.g., 64,128,256): ').strip())
    print('Training codebook (this may take time)...')

    cb = Codebook(vector_size=samples.shape[1])
    start = time.time()
    cb.lbg_training(samples, k)
    duration = time.time() - start
    print(f'Codebook trained in {duration:.2f}s. Codebook size: {len(cb.codewords)}')

    indices = cb.encode(samples)

    # save outputs
    base = os.path.splitext(os.path.basename(path))[0]
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_prefix = f'{base}_VQ_{timestamp}'
    codebook_path = out_prefix + '_codebook.json'
    compressed_path = out_prefix + '_compressed.npz'
    reconstructed_path = out_prefix + '_reconstructed.png'

    cb.save(codebook_path)
    np.savez_compressed(compressed_path, indices=indices, shape=arr.shape, pad=pad_info, block=(bh,bw), codebook_size=len(cb.codewords), version_timestamp=timestamp)

    # reconstruct for quick verify
    reconstructed_vectors = cb.decode(indices)
    reconstructed_img = image_from_blocks(reconstructed_vectors, padded.shape, bh, bw)
    # remove padding
    h, w = arr.shape
    reconstructed_cropped = reconstructed_img[:h, :w]
    save_image_from_array(reconstructed_cropped, reconstructed_path)

    # compute compression ratio estimate
    orig_bytes, comp_bytes = compute_sizes(arr, bh, bw, len(cb.codewords))
    ratio = orig_bytes / comp_bytes if comp_bytes>0 else float('inf')

    # Save metadata including release/version date (current time) -- assignment needs release/version date requested before discussion
    metadata = {
        'original_file': path,
        'timestamp_utc': timestamp,
        'release_version': timestamp,
        'block_size': (bh,bw),
        'padding': pad_info,
        'codebook_size': len(cb.codewords),
        'reconstructed_image': reconstructed_path,
        'codebook_file': codebook_path,
        'compressed_file': compressed_path,
        'compression_ratio_estimated': ratio
    }
    with open(out_prefix + '_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print('\nCompression complete!')
    print('Saved:')
    print(' - Codebook ->', codebook_path)
    print(' - Compressed indices ->', compressed_path)
    print(' - Reconstructed image (for verification) ->', reconstructed_path)
    print(' - Metadata ->', out_prefix + '_metadata.json')
    print(f'Estimated compression ratio (orig_bytes / compressed_bytes) = {ratio:.2f}')


def decompress_flow():
    codebook_path = input('Enter codebook file path (json): ').strip()
    compressed_path = input('Enter compressed file path (.npz): ').strip()
    if not os.path.isfile(codebook_path) or not os.path.isfile(compressed_path):
        print('One or both files not found.'); return

    cb = Codebook.load(codebook_path)
    data = np.load(compressed_path)
    indices = data['indices']
    shape = tuple(data['shape'].tolist()) if hasattr(data['shape'], 'tolist') else tuple(data['shape'])
    pad = tuple(data['pad'].tolist()) if hasattr(data['pad'], 'tolist') else tuple(data['pad'])
    block = tuple(data['block'].tolist()) if hasattr(data['block'], 'tolist') else tuple(data['block'])

    bh, bw = block
    # decode
    vectors = cb.decode(indices)
    padded_h = shape[0] + pad[0]
    padded_w = shape[1] + pad[1]
    reconstructed_img = image_from_blocks(vectors, (padded_h, padded_w), bh, bw)
    reconstructed_cropped = reconstructed_img[:shape[0], :shape[1]]

    out_path = os.path.splitext(os.path.basename(compressed_path))[0] + '_decompressed.png'
    save_image_from_array(reconstructed_cropped, out_path)
    print('Decompressed image saved to:', out_path)


def main_menu():
    print('=== Vector Quantizer (VQ) Assignment Tool ===')
    while True:
        print('\nMenu:')
        print('1) Compress image')
        print('2) Decompress image')
        print('3) Exit')
        choice = input('Enter choice: ').strip()
        if choice == '1':
            compress_flow()
        elif choice == '2':
            decompress_flow()
        elif choice == '3':
            print('Exit. Good luck with submission!')
            break
        else:
            print('Invalid choice. Try again.')


if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print('\nInterrupted. Exiting.')
