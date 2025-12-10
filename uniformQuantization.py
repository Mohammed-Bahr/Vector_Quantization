import numpy as np
import json
from PIL import Image


def generate_codebook_uniform(target, bits=2, codebook_json="codebook.json", codebook_txt="codebook.txt",
                              global_min=0, global_max=255):
    
    if bits <= 0:
        raise ValueError("bits must be >= 1")

    L = 2 ** bits
    total_values = int((global_max - global_min) + 1)  # 256

    step = float(total_values / L)
    
    rmins = []
    rmaxs = []
    cur = float(global_min)
    for i in range(L):
        rmin = cur
        rmax = cur + step - 1
        rmins.append(float(rmin))
        rmaxs.append(float(rmax))
        cur = rmax + 1

    # compute midpoints
    midpoints = [ (rmins[i] + rmaxs[i]) / 2.0 for i in range(L) ]

    json_list = []
    for i in range(L):
        json_list.append({
            "code": int(i),
            "midpoint": float(midpoints[i]),
            "range": [float(rmins[i]), float(rmaxs[i])]
        })

    # save JSON
    with open(codebook_json, "w") as f:
        json.dump(json_list, f, indent=4)
    print(f"Codebook JSON saved: {codebook_json}")

    # save TXT summary
    with open(codebook_txt, "w") as f:
        f.write(f"{'Level':<6}{'Midpoint':>12}{'RangeMin':>12}{'RangeMax':>12}\n")
        f.write("-" * 50 + "\n")
        for i in range(L):
            f.write(f"{i:<6}{midpoints[i]:>12.2f}{rmins[i]:>12}{rmaxs[i]:>12}\n")
    print(f"Codebook TXT saved: {codebook_txt}")



def main():
    try:
        img = Image.open("leaf.jpg").convert("L")  # convert it to grayscale 
    except FileNotFoundError:
        print("Error: leaf.jpg not found!")
        return

    img_array = np.array(img)

    bits = 2  

    generate_codebook_uniform(img_array, bits=bits)
    print("Uniform quantization codebook generated.")

if __name__ == "__main__":
    main()

