import json
import numpy as np
from PIL import Image, ImageTk
from scipy.spatial.distance import cdist
import os
import math
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))  # script folder

# =========================================================
# CODEBOOK CLASS (GRAYSCALE + COLOR SUPPORT)
# =========================================================
class Codebook:
    def __init__(self, path, block_h, block_w):
        self.path = path
        self.block_h = block_h
        self.block_w = block_w

        img = Image.open(self.path).convert("RGB")
        self.img_arr = np.array(img)
        self.orig_h, self.orig_w, self.channels = self.img_arr.shape

        pad_h = (self.block_h - (self.orig_h % self.block_h)) % self.block_h
        pad_w = (self.block_w - (self.orig_w % self.block_w)) % self.block_w

        self.img_padded = np.pad(
            self.img_arr,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0
        )

        self.padded_h, self.padded_w, _ = self.img_padded.shape
        self.blocks = self.image_to_blocks()
        self.codebook = None
        self.n_rows = self.padded_h // self.block_h
        self.n_cols = self.padded_w // self.block_w

        # Output files
        self.base_name = os.path.splitext(os.path.basename(self.path))[0]
        self.codebook_json = os.path.join(script_dir, f"{self.base_name}_codebook.json")
        self.codebook_txt = os.path.join(script_dir, f"{self.base_name}_codebook.txt")
        self.labels_json = os.path.join(script_dir, f"{self.base_name}_labels.json")
        self.labels_bin = os.path.join(script_dir, f"{self.base_name}_labels.bin")
        self.reconstructed_path = os.path.join(script_dir, f"{self.base_name}_reconstructed.png")

    # -----------------------------------------------------
    def image_to_blocks(self):
        h, w, c = self.img_padded.shape
        n_rows = h // self.block_h
        n_cols = w // self.block_w
        blocks = self.img_padded.reshape(n_rows, self.block_h, n_cols, self.block_w, c)
        blocks = blocks.swapaxes(1, 2)
        return blocks.reshape(-1, self.block_h * self.block_w * c)

    # -----------------------------------------------------
    def generate_codebook(self, k, epsilon=0.01, threshold=0.001, max_iterations=100):
        if k > len(self.blocks):
            raise ValueError(
                f"Invalid quantization level k={k}: cannot exceed the total number of image blocks ({len(self.blocks)})."
            )

        print(f"\n=== Starting LBG for k={k} ===")
        centroid = np.mean(self.blocks, axis=0)
        self.codebook = np.array([centroid])

        while len(self.codebook) < k:
            code_plus = self.codebook * (1 + epsilon)
            code_minus = self.codebook * (1 - epsilon)
            self.codebook = np.vstack((code_plus, code_minus))

            prev_distortion = float('inf')
            for i in range(max_iterations):
                distances = cdist(self.blocks, self.codebook, metric='cityblock')
                labels = np.argmin(distances, axis=1)
                new_codebook = np.zeros_like(self.codebook)

                for idx in range(len(self.codebook)):
                    members = self.blocks[labels == idx]
                    if len(members) > 0:
                        new_codebook[idx] = np.mean(members, axis=0)
                    else:
                        new_codebook[idx] = self.codebook[idx]

                self.codebook = new_codebook
                min_distances = distances[np.arange(len(distances)), labels]
                distortion = np.mean(min_distances)

                if prev_distortion != float('inf'):
                    change = abs(prev_distortion - distortion) / prev_distortion
                    if change < threshold:
                        print(f"Converged at iter {i}, distortion={distortion:.3f}")
                        break
                
                # Optional: print progress every few iterations if not converging yet
                if i % 10 == 0 and i > 0:
                     print(f"Iteration {i}, distortion={distortion:.3f}")

                prev_distortion = distortion

        # Save codebook as JSON
        final = self.codebook.reshape(-1, self.block_h, self.block_w, self.channels).tolist()
        with open(self.codebook_json, "w") as f:
            json.dump(final, f, indent=4)
        print(f"✓ Codebook saved to JSON: {self.codebook_json}")

        # Save codebook as TXT table
        with open(self.codebook_txt, "w") as f:
            f.write(f"{'Level':<6}{'Min':>10}{'Max':>10}{'Dequantized':>30}\n")
            f.write("-"*60 + "\n")
            for idx, vec in enumerate(self.codebook):
                min_val = vec.min()
                max_val = vec.max()
                dequant_val = np.round(vec.mean(), 2)
                f.write(f"{idx:<6}{min_val:>10.2f}{max_val:>10.2f}{dequant_val:>30.2f}\n")
        print(f"✓ Codebook saved as formatted TXT: {self.codebook_txt}")

        return final

    # -----------------------------------------------------
    def compress(self):
        if self.codebook is None:
            raise ValueError("No codebook yet.")

        distances = cdist(self.blocks, self.codebook, metric="cityblock")
        labels = np.argmin(distances, axis=1)
        labels_grid = labels.reshape(self.n_rows, self.n_cols)

        # Save labels as JSON
        with open(self.labels_json, "w") as f:
            json.dump(labels_grid.tolist(), f)
        print(f"✓ Labels saved as JSON: {self.labels_json}")

        # Save labels as binary
        bits_needed = math.ceil(math.log2(len(self.codebook)))
        binary_data = 0
        bit_count = 0
        with open(self.labels_bin, "wb") as f:
            for lbl in labels.flatten():
                binary_data = (binary_data << bits_needed) | lbl
                bit_count += bits_needed
                while bit_count >= 8:
                    byte = (binary_data >> (bit_count - 8)) & 0xFF
                    f.write(struct.pack("B", byte))
                    bit_count -= 8
            if bit_count > 0:
                byte = (binary_data << (8 - bit_count)) & 0xFF
                f.write(struct.pack("B", byte))
        print(f"✓ Labels saved as binary: {self.labels_bin}")

        return labels_grid

    # -----------------------------------------------------
    @staticmethod
    def decompress(labels_path, codebook_path, output_path):
        labels = np.array(json.load(open(labels_path)))
        codebook = np.array(json.load(open(codebook_path)))

        n_rows, n_cols = labels.shape
        block_h, block_w, channels = codebook.shape[1], codebook.shape[2], codebook.shape[3]

        img_rows = []
        for r in range(n_rows):
            block_row = [codebook[labels[r][c]] for c in range(n_cols)]
            for i in range(block_h):
                row = []
                for blk in block_row:
                    row.extend(blk[i])
                img_rows.append(row)

        arr = np.array(img_rows, dtype=np.uint8).reshape(n_rows*block_h, n_cols*block_w, channels)
        Image.fromarray(arr, "RGB").save(output_path)
        print(f"✓ Decompression done. Saved as {output_path}")
        return arr

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

# =========================================================
# GUI UTILS & CLASS
# =========================================================

class TextRedirector:
    """Redirects text to a tkinter Text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        # Use after to ensure thread safety with Tkinter
        try:
            self.widget.after(0, self._write, str)
        except:
            pass

    def _write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass

class VQ_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vector Quantization System")
        self.root.geometry("950x650")
        
        # --- THEME & STYLE ---
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Colors
        self.bg_color = "#f4f4f4"
        self.panel_bg = "#ffffff"
        self.accent_color = "#0078d7"
        
        self.root.configure(bg=self.bg_color)
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 9, "bold"))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#333")
        self.style.configure("Card.TFrame", background=self.panel_bg, relief="flat")
        
        # --- LAYOUT ---
        # Split window: Left (Controls) | Right (Logs)
        self.paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.left_panel = ttk.Frame(self.paned)
        self.right_panel = ttk.Frame(self.paned)
        
        self.paned.add(self.left_panel, weight=2)
        self.paned.add(self.right_panel, weight=3)
        
        # --- LEFT PANEL: CONTROLS ---
        self.setup_header()
        self.setup_notebook()
        
        # --- RIGHT PANEL: LOGS ---
        self.setup_log_area()
        
        # --- REDIRECT STDOUT ---
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")
        
        print("Welcome to Vector Quantization System.")
        print("Ready for operations...\n")

    def setup_header(self):
        header_frame = ttk.Frame(self.left_panel)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        lbl = ttk.Label(header_frame, text="VQ System", style="Header.TLabel")
        lbl.pack(side=tk.LEFT)

    def setup_notebook(self):
        self.notebook = ttk.Notebook(self.left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_compress = ttk.Frame(self.notebook, padding=15)
        self.tab_decompress = ttk.Frame(self.notebook, padding=15)
        
        self.notebook.add(self.tab_compress, text="  Compress  ")
        self.notebook.add(self.tab_decompress, text="  Decompress  ")
        
        self.setup_compress_tab()
        self.setup_decompress_tab()

    def setup_compress_tab(self):
        frame = self.tab_compress
        
        # -- Section: Input --
        grp_input = ttk.LabelFrame(frame, text=" Input Image ", padding=10)
        grp_input.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(grp_input, text="Path:").pack(anchor="w")
        
        f_path = ttk.Frame(grp_input)
        f_path.pack(fill=tk.X, pady=5)
        
        self.entry_img_path = ttk.Entry(f_path)
        self.entry_img_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(f_path, text="Browse", command=self.browse_image).pack(side=tk.RIGHT)

        # -- Section: Settings --
        grp_settings = ttk.LabelFrame(frame, text=" Parameters ", padding=10)
        grp_settings.pack(fill=tk.X, pady=(0, 15))
        
        # Grid for settings
        grp_settings.columnconfigure(1, weight=1)
        
        ttk.Label(grp_settings, text="Block Height:").grid(row=0, column=0, sticky="w", pady=5)
        self.entry_bh = ttk.Entry(grp_settings, width=10)
        self.entry_bh.insert(0, "4")
        self.entry_bh.grid(row=0, column=1, sticky="w", padx=10)
        
        ttk.Label(grp_settings, text="Block Width:").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_bw = ttk.Entry(grp_settings, width=10)
        self.entry_bw.insert(0, "4")
        self.entry_bw.grid(row=1, column=1, sticky="w", padx=10)
        
        ttk.Label(grp_settings, text="Quantization Levels (k):").grid(row=2, column=0, sticky="w", pady=5)
        self.entry_k = ttk.Entry(grp_settings, width=10)
        self.entry_k.insert(0, "16")
        self.entry_k.grid(row=2, column=1, sticky="w", padx=10)
        
        # -- Section: Action --
        self.btn_compress = ttk.Button(frame, text="START COMPRESSION", command=self.start_compression_thread)
        self.btn_compress.pack(fill=tk.X, pady=10)

    def setup_decompress_tab(self):
        frame = self.tab_decompress
        
        grp_files = ttk.LabelFrame(frame, text=" Input Files ", padding=10)
        grp_files.pack(fill=tk.X, pady=(0, 15))
        
        # Labels File
        ttk.Label(grp_files, text="Labels JSON:").pack(anchor="w")
        f_lbl = ttk.Frame(grp_files)
        f_lbl.pack(fill=tk.X, pady=5)
        self.entry_labels = ttk.Entry(f_lbl)
        self.entry_labels.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(f_lbl, text="Browse", command=lambda: self.browse_file(self.entry_labels, "json")).pack(side=tk.RIGHT)
        
        # Codebook File
        ttk.Label(grp_files, text="Codebook JSON:").pack(anchor="w", pady=(10, 0))
        f_cb = ttk.Frame(grp_files)
        f_cb.pack(fill=tk.X, pady=5)
        self.entry_codebook = ttk.Entry(f_cb)
        self.entry_codebook.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(f_cb, text="Browse", command=lambda: self.browse_file(self.entry_codebook, "json")).pack(side=tk.RIGHT)
        
        # -- Section: Action --
        self.btn_decompress = ttk.Button(frame, text="START DECOMPRESSION", command=self.start_decompression_thread)
        self.btn_decompress.pack(fill=tk.X, pady=10)

    def setup_log_area(self):
        lbl_frame = ttk.Frame(self.right_panel)
        lbl_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(lbl_frame, text="Console Output", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(lbl_frame, text="Clear Log", width=10, command=self.clear_log).pack(side=tk.RIGHT)
        
        self.log_text = scrolledtext.ScrolledText(
            self.right_panel, 
            state='disabled', 
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Tags for coloring
        self.log_text.tag_config("stdout", foreground="#d4d4d4")
        self.log_text.tag_config("stderr", foreground="#ff6b6b")
        self.log_text.tag_config("success", foreground="#4ec9b0")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")

    def browse_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if filename:
            self.entry_img_path.delete(0, tk.END)
            self.entry_img_path.insert(0, filename)

    def browse_file(self, entry_widget, ext):
        filename = filedialog.askopenfilename(filetypes=[(f"{ext.upper()} files", f"*.{ext}")])
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)

    # --- THREADING HELPERS ---
    def start_compression_thread(self):
        t = threading.Thread(target=self.run_compression)
        t.daemon = True
        t.start()

    def start_decompression_thread(self):
        t = threading.Thread(target=self.run_decompression)
        t.daemon = True
        t.start()

    def run_compression(self):
        path = self.entry_img_path.get()
        try:
            bh = int(self.entry_bh.get())
            bw = int(self.entry_bw.get())
            k = int(self.entry_k.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for parameters.")
            return

        if not os.path.exists(path):
            messagebox.showerror("Error", "Image file not found.")
            return

        self.btn_compress.config(state="disabled")
        print("\n--- Starting Compression Process ---")
        
        try:
            cb = Codebook(path, bh, bw)
            cb.generate_codebook(k)
            cb.compress()
            print("--- Compression Completed Successfully ---")
            messagebox.showinfo("Success", "Compression completed!")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.btn_compress.config(state="normal")

    def run_decompression(self):
        labels_path = self.entry_labels.get()
        codebook_path = self.entry_codebook.get()
        
        if not os.path.exists(labels_path) or not os.path.exists(codebook_path):
            messagebox.showerror("Error", "Please select valid files.")
            return

        # Infer output name based on labels file name
        base_name = os.path.splitext(os.path.basename(labels_path))[0].replace("_labels", "")
        output_path = os.path.join(script_dir, f"{base_name}_reconstructed_gui.png")

        self.btn_decompress.config(state="disabled")
        print("\n--- Starting Decompression Process ---")

        try:
            Codebook.decompress(labels_path, codebook_path, output_path)
            print("--- Decompression Completed Successfully ---")
            messagebox.showinfo("Success", f"Decompression completed.\nSaved to: {output_path}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.btn_decompress.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = VQ_GUI(root)
    root.mainloop()