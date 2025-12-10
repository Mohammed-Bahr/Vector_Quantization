# import numpy as np
# import json
# from PIL import Image


# def generate_codebook_uniform(target, bits=2, codebook_json="codebook.json", codebook_txt="codebook.txt",
#                               global_min=0, global_max=255):
#     """
#     Build a uniform scalar quantizer codebook with integer, non-overlapping ranges.
#     Ranges cover [global_min .. global_max] inclusive and are split into L = 2**bits bins.
#     Each JSON entry: { "code": i, "midpoint": <float>, "range": [rmin, rmax] }.
#     """
#     if bits <= 0:
#         raise ValueError("bits must be >= 1")

#     L = 2 ** bits
#     total_values = int(global_max - global_min + 1)  # typically 256 for 0..255

#     # integer bin sizes: distribute remainder to first bins to keep contiguous non-overlapping ranges
#     base_size = total_values // L
#     remainder = total_values % L
#     sizes = [base_size + (1 if i < remainder else 0) for i in range(L)]

#     # build rmin/rmax
#     rmins = []
#     rmaxs = []
#     cur = int(global_min)
#     for s in sizes:
#         rmin = cur
#         rmax = cur + s - 1
#         rmins.append(int(rmin))
#         rmaxs.append(int(rmax))
#         cur = rmax + 1

#     # compute midpoints
#     midpoints = [ (rmins[i] + rmaxs[i]) / 2.0 for i in range(L) ]

#     # prepare JSON list
#     json_list = []
#     for i in range(L):
#         json_list.append({
#             "code": int(i),
#             "midpoint": float(midpoints[i]),
#             "range": [int(rmins[i]), int(rmaxs[i])]
#         })

#     # save JSON
#     with open(codebook_json, "w") as f:
#         json.dump(json_list, f, indent=4)
#     print(f"Codebook JSON saved: {codebook_json}")

#     # save TXT summary
#     with open(codebook_txt, "w") as f:
#         f.write(f"{'Level':<6}{'Midpoint':>12}{'RangeMin':>12}{'RangeMax':>12}\n")
#         f.write("-" * 50 + "\n")
#         for i in range(L):
#             f.write(f"{i:<6}{midpoints[i]:>12.2f}{rmins[i]:>12}{rmaxs[i]:>12}\n")
#     print(f"Codebook TXT saved: {codebook_txt}")

#     return {
#         "json": json_list,
#         "midpoints": midpoints,
#         "rmins": rmins,
#         "rmaxs": rmaxs,
#         "sizes": sizes
#     }

# def main():
#     try:
#         img = Image.open("leaf.jpg").convert("L")  # convert it to grayscale 
#     except FileNotFoundError:
#         print("Error: leaf.jpg not found!")
#         return

#     img_array = np.array(img)

#     bits = 2  

#     # produce integer, non-overlapping ranges in JSON/TXT
#     generate_codebook_uniform(img_array, bits=bits, codebook_json="uniform_codebook.json", codebook_txt="uniform_codebook.txt")

# if __name__ == "__main__":
#     main()


import numpy as np
import json
from PIL import Image


def generate_uniform_codebook(bits, global_min=0, global_max=255):
    """ Generates a 1D uniform scalar quantizer codebook """
    L = 2 ** bits
    total_values = int(global_max - global_min + 1)
    step = total_values / L

    ranges = []
    midpoints = []

    cur = float(global_min)
    for _ in range(L):
        rmin = cur
        rmax = cur + step - 1
        ranges.append((rmin, rmax))
        midpoints.append((rmin + rmax) / 2.0)
        cur = rmax + 1

    return ranges, midpoints


def quantize_channel(channel, ranges, midpoints):
    """Quantizes a single color channel"""
    flat = channel.flatten()
    quantized = np.zeros_like(flat, dtype=np.uint8)

    for i, value in enumerate(flat):
        for idx, (rmin, rmax) in enumerate(ranges):
            if rmin <= value <= rmax:
                quantized[i] = int(midpoints[idx])
                break

    return quantized.reshape(channel.shape)


def quantize_rgb_image(img_array, bits):
    """Quantize an RGB image channel by channel."""

    # Split R, G, B
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]

    ranges, midpoints = generate_uniform_codebook(bits)

    # Quantize each channel
    Rq = quantize_channel(R, ranges, midpoints)
    Gq = quantize_channel(G, ranges, midpoints)
    Bq = quantize_channel(B, ranges, midpoints)

    # Combine back
    quantized_img = np.stack([Rq, Gq, Bq], axis=2).astype(np.uint8)

    return quantized_img, ranges, midpoints


def save_codebook(ranges, midpoints, filename_json="codebook_rgb.json", filename_txt="codebook_rgb.txt"):
    """Save the codebook to JSON & TXT"""
    codebook = []
    for i in range(len(midpoints)):
        codebook.append({
            "level": i,
            "midpoint": midpoints[i],
            "range": [ranges[i][0], ranges[i][1]]
        })

    with open(filename_json, "w") as f:
        json.dump(codebook, f, indent=4)

    with open(filename_txt, "w") as f:
        f.write(f"{'Level':<6}{'Midpoint':>12}{'Min':>12}{'Max':>12}\n")
        f.write("-"*50 + "\n")
        for i in range(len(midpoints)):
            f.write(f"{i:<6}{midpoints[i]:>12.2f}{ranges[i][0]:>12.2f}{ranges[i][1]:>12.2f}\n")

    print("Codebook saved (JSON & TXT).")


def main():
    try:
        img = Image.open("leaf.jpg")  # RGB image
    except FileNotFoundError:
        print("Error: leaf.jpg not found!")
        return

    img_array = np.array(img)

    bits = 2

    quantized_img, ranges, midpoints = quantize_rgb_image(img_array, bits)

    # Save quantized image
    Image.fromarray(quantized_img).save("quantized_leaf.jpg")
    print("Quantized image saved as quantized_leaf.jpg")

    # Save codebook
    save_codebook(ranges, midpoints)

    print("RGB uniform quantization completed.")


if __name__ == "__main__":
    main()
