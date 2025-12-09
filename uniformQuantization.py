import numpy as np
import json
from PIL import Image


def generate_codebook_uniform(target, bits=2):

    codebook_json="codebook.json"
    codebook_txt="codebook.txt"
    global_min=0.0
    global_max=255.0
    block_h=1
    block_w=1
    channels=1

    #------------------------------------------
    if bits <= 0:
        raise ValueError("bits must be >= 1")

    L = 2 ** bits
    Delta = (global_max - global_min) / L


    #==========================================
        # midpoints for each level

    # def simple_uniform_codebook(bits, min_val, max_val):
    #     L = 2 ** bits            
    #     Delta = (max_val - min_val) / L    
        
    #     midpoints = []
        
    #     for i in range(L):
    #         rmin = min_val + i * Delta
    #         rmax = min_val + (i + 1) * Delta
    #         midpoint = (rmin + rmax) / 2
            
    #         midpoints.append(midpoint)
        
    #     return midpoints

    # ==========================================
    #    | 
    #   \/ like arrow hahhahahha
    # this bellow line summerize the above commented function
    midpoints = (global_min + (np.arange(L) + 0.5) * Delta).astype(float)

    
    json_list = []
    for i in range(L):
        rmin = float(global_min + i * Delta)
        rmax = float(global_min + (i + 1) * Delta)
        json_list.append({
            "code": int(i),
            "midpoint": float(midpoints[i]),
            "range": [rmin, rmax]
        })

    with open(codebook_json, "w") as f:
        json.dump(json_list, f, indent=4)
    print(f"Codebook JSON saved: {codebook_json}")

    # Save TXT summary
    with open(codebook_txt, "w") as f:
        f.write(f"{'Level':<6}{'Midpoint':>12}{'RangeMin':>12}{'RangeMax':>12}\n")
        f.write("-" * 50 + "\n")
        for i in range(L):
            rmin = global_min + (i * Delta)
            rmax = global_min + ((i + 1) * Delta)
            f.write(f"{i:<6}{midpoints[i]:>12.2f}{rmin:>12.2f}{rmax:>12.2f}\n")
    print(f"Codebook TXT saved: {codebook_txt}")

    


def main():
    try:
        img = Image.open("leaf.jpg").convert("L")  # convert it to grayscale 
    except FileNotFoundError:
        print("Error: leaf.jpg not found!")
        return

    img_array = np.array(img)

    bits = 2  


    generate_codebook_uniform(img_array)




if __name__ == "__main__":
    main()

