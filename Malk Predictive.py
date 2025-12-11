import json
import numpy as np
from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

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

def loco_predict(img, i, j, c):
    if i == 0 or j == 0:
        return int(img[i, j, c])
    A = int(img[i, j-1, c])  # Left
    B = int(img[i-1, j, c])  # Top
    C = int(img[i-1, j-1, c])  # Top-left
    if C >= max(A, B):
        return min(A, B)
    elif C <= min(A, B):
        return max(A, B)
    else:
        return A + B - C

def analysis_pass(image_path):
    img = np.array(Image.open(image_path).convert('RGB'), dtype=np.int32)
    h, w, _ = img.shape
    global_min = [999999]*3
    global_max = [-999999]*3
    for i in range(h):
        for j in range(w):
            for c in range(3):
                pred = loco_predict(img, i, j, c)
                err = img[i, j, c] - pred
                if err < global_min[c]: global_min[c] = err
                if err > global_max[c]: global_max[c] = err
    return global_min, global_max

def generate_codebook_uniform_rgb(basename,bits=2, codebook_json="codebook_rgb.json", codebook_txt="codebook_rgb.txt", global_mins=(0,0,0), global_maxs=(255,255,255)):
    if bits <= 0:
        raise ValueError("bits must be >= 1")
    L = 2 ** bits
    channels = ['R','G','B']
    codebooks = {}
    for idx, ch in enumerate(channels):
        gmin = global_mins[idx]
        gmax = global_maxs[idx]
        total_values = gmax - gmin + 1
        step = float(total_values / L)
        rmins, rmaxs = [], []
        cur = float(gmin)
        for i in range(L):
            rmin, rmax = cur, cur + step - 1
            rmins.append(rmin); rmaxs.append(rmax)
            cur = rmax + 1
        midpoints = [(rmins[i]+rmaxs[i])/2.0 for i in range(L)]
        channel_list = [{"code": i, "midpoint": midpoints[i], "range": [rmins[i], rmaxs[i]]} for i in range(L)]
        codebooks[ch] = channel_list

    codebook_json = os.path.join(script_dir, basename + codebook_json)
    with open(codebook_json, "w") as f:
        json.dump(codebooks, f, indent=4)
    
    codebook_txt = os.path.join(script_dir, basename + codebook_txt)
    with open(codebook_txt, "w") as f:
        for ch in channels:
            f.write(f"Channel: {ch}\n")
            f.write(f"{'Level':<6}{'Midpoint':>12}{'RangeMin':>12}{'RangeMax':>12}\n")
            f.write("-"*50 + "\n")
            for entry in codebooks[ch]:
                f.write(f"{entry['code']:<6}{entry['midpoint']:>12.2f}{entry['range'][0]:>12}{entry['range'][1]:>12}\n")
            f.write("\n")
    print(f"Codebooks saved: {codebook_json}, {codebook_txt}")

def find_quant_index(err, codebook):
    for entry in codebook:
        rmin, rmax = entry['range']
        if rmin <= err <= rmax:
            return entry['code']
    return 0 if err < codebook[0]['range'][0] else codebook[-1]['code']

def compress_rgb(original_img, codebook_json):
    h, w, _ = original_img.shape
    reconstructed = np.zeros_like(original_img, dtype=np.int32)
    quant_indices = np.zeros((h, w, 3), dtype=np.int32)

    predicted = np.zeros_like(original_img, dtype=np.int32)
    error = np.zeros_like(original_img, dtype=np.int32)
    q_image = np.zeros_like(original_img, dtype=np.int32)

    with open(codebook_json, "r") as f:
        codebooks = json.load(f)
    channels = ['R','G','B']

    for i in range(h):
        for j in range(w):
            for c_idx, ch in enumerate(channels):
                codebook = codebooks[ch]
                pred = loco_predict(reconstructed, i, j, c_idx)
                err = original_img[i, j, c_idx] - pred
                q_index = find_quant_index(err, codebook)
                dq_err = codebook[q_index]['midpoint']
                recon_pixel = pred + dq_err
                recon_pixel = max(0, min(255, int(round(recon_pixel))))
                reconstructed[i, j, c_idx] = recon_pixel
                quant_indices[i, j, c_idx] = q_index

               
                predicted[i,j,c_idx] = pred
                error[i,j,c_idx] = err
                q_image[i,j,c_idx] = dq_err

    return reconstructed, quant_indices, predicted, error, q_image

def save_quantized_bin(basename, quant_indices):
    h, w, _ = quant_indices.shape
    bin_path = os.path.join(script_dir, f"{basename}_quant.bin")

    with open(bin_path, "wb") as f:
        # Write header
        f.write(np.int32(h).tobytes())
        f.write(np.int32(w).tobytes())
        
        # Flatten indices and write as uint8
        flat = quant_indices.astype(np.uint8).flatten()
        f.write(flat.tobytes())

    print(f"Quantized indices saved to binary: {bin_path}")


def save_images(basename, predicted, error, quant_indices, q_image, reconstructed):
    # Predicted image
    Image.fromarray(
        np.clip(predicted, 0, 255).astype(np.uint8)
    ).save(os.path.join(script_dir, f"{basename}_predicted.png"))

    # Original prediction error (shift +128)
    Image.fromarray(
        np.clip(error + 128, 0, 255).astype(np.uint8)
    ).save(os.path.join(script_dir, f"{basename}_error.png"))

    # Quantized error indices (shifted for visualization)
    Image.fromarray(
        np.clip(quant_indices + 128, 0, 255).astype(np.uint8)
    ).save(os.path.join(script_dir, f"{basename}_quantized_error.png"))

    # Dequantized error values (also shifted for visualization)
    Image.fromarray(
        np.clip(q_image + 128, 0, 255).astype(np.uint8)
    ).save(os.path.join(script_dir, f"{basename}_dequantized_error.png"))

    # Reconstructed image
    Image.fromarray(
        np.clip(reconstructed, 0, 255).astype(np.uint8)
    ).save(os.path.join(script_dir, f"{basename}_reconstructed.png"))

    print(
        f"All images saved:\n"
        f" - {basename}_predicted.png\n"
        f" - {basename}_error.png\n"
        f" - {basename}_quantized_error.png\n"
        f" - {basename}_dequantized_error.png\n"
        f" - {basename}_reconstructed.png"
    )
    
def decompress_rgb(basename, codebook_json):
    # Load quantized error indices (shift removed)
    q_path = os.path.join(script_dir, f"{basename}_quantized_error.png")
    q_img = np.array(Image.open(q_path), dtype=np.int32)

    # Remove visualization shift (+128)
    quant_indices = q_img - 128

    # Load codebook
    with open(codebook_json, "r") as f:
        codebooks = json.load(f)

    channels = ['R', 'G', 'B']
    h, w, _ = quant_indices.shape

    reconstructed = np.zeros((h, w, 3), dtype=np.int32)

    # Predictive decoding
    for i in range(h):
        for j in range(w):
            for c_idx, ch in enumerate(channels):

                # Predictor uses reconstructed image (feedback loop)
                pred = loco_predict(reconstructed, i, j, c_idx)

                # Get quantized error index
                q_index = int(quant_indices[i, j, c_idx])

                # Clamp index to valid range
                q_index = max(0, min(q_index, len(codebooks[ch]) - 1))

                # Get midpoint (dequantized error)
                dq_err = float(codebooks[ch][q_index]["midpoint"])

                # Reconstruct pixel
                recon_pixel = pred + dq_err

                # Clip to valid range
                recon_pixel = max(0, min(255, int(round(recon_pixel))))

                reconstructed[i, j, c_idx] = recon_pixel

    return reconstructed



# --- Main Menu ---
if __name__ == "__main__":
    while True:
        print("\n=== Predictive Coding Menu ===")
        print("1. Compress Image")
        print("2. Decompress Image")
        print("3. Exit")
        choice = input("Enter choice [1-3]: ").strip()

        if choice == "1":
            image_path = input("Enter image path: ").strip()
            try:
                image_path = validate_image_path(image_path)
            except Exception as e:
                print(f"Error: {e}")
                continue

            basename = os.path.splitext(os.path.basename(image_path))[0]

            num_bits = input("Enter number of bits for quantization (e.g., 2): ").strip()
            try:
                num_bits = int(num_bits)
                if num_bits <= 0:
                    raise ValueError
            except:
                print("Invalid number of bits.")
                continue

            print("Running analysis pass...")
            global_min, global_max = analysis_pass(image_path)
            print("Global min errors:", [int(x) for x in global_min])
            print("Global max errors:", [int(x) for x in global_max])

            print("Generating codebooks...")
            generate_codebook_uniform_rgb(
                basename=basename,
                bits=num_bits,
                global_mins=tuple(global_min),
                global_maxs=tuple(global_max)
            )

            img = np.array(Image.open(image_path).convert('RGB'), dtype=np.int32)
            print("Running compression pass...")

            codebook_path = os.path.join(script_dir, basename + "codebook_rgb.json")

            reconstructed, quant_indices, predicted, error, q_image = compress_rgb(
                img, codebook_path
            )

            save_quantized_bin(basename, quant_indices)
            save_images(basename, predicted, error, quant_indices, q_image, reconstructed)



        elif choice == "2":
            basename = input("Enter image basename (without extension): ").strip()

            codebook_file = basename + "codebook_rgb.json"
            codebook_path = os.path.join(script_dir, codebook_file)

            if not os.path.exists(codebook_path):
                print("Error: Codebook not found. Run compression first.")
                continue

            print("Running decompression...")
            reconstructed = decompress_rgb(basename, codebook_path)

            out_path = os.path.join(script_dir, f"{basename}_decompressed.png")
            Image.fromarray(reconstructed.astype(np.uint8)).save(out_path)

            print(f"Decompressed image saved as: {basename}_decompressed.png")


        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice.")
