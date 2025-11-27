import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import os
import json
import numpy as np
from scipy.spatial.distance import cdist
import math
import struct
import threading
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# CODEBOOK CLASS (Copied from GUI_VQ_Full.py)
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

    def image_to_blocks(self):
        h, w, c = self.img_padded.shape
        n_rows = h // self.block_h
        n_cols = w // self.block_w
        blocks = self.img_padded.reshape(n_rows, self.block_h, n_cols, self.block_w, c)
        blocks = blocks.swapaxes(1, 2)
        return blocks.reshape(-1, self.block_h * self.block_w * c)

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

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, radius=20, bg="#4a6fc7", fg="white", 
                 font=("Arial", 10, "bold"), width=120, height=35, **kwargs):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg=parent.cget('bg'))
        self.command = command
        self.radius = radius
        self.bg = bg
        self.fg = fg
        self.font = font
        self.width = width
        self.height = height
        self.text = text
        

        # binding of events like clicking on a button and hover effects
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        self.draw_button()
        
    def draw_button(self, hover=False):
        self.delete("all")
        color = self.bg
        if hover:
            # Lighten color on hover
            r, g, b = int(self.bg[1:3], 16), int(self.bg[3:5], 16), int(self.bg[5:7], 16)
            r = min(255, int(r * 1.2))
            g = min(255, int(g * 1.2))
            b = min(255, int(b * 1.2))
            color = f"#{r:02x}{g:02x}{b:02x}"
        
        # Draw rounded rectangle
        self.create_rounded_rectangle(0, 0, self.width, self.height, radius=self.radius, fill=color, outline="")
        
        # Add text
        self.create_text(self.width//2, self.height//2, text=self.text, fill=self.fg, font=self.font)
    
    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _on_click(self, event):
        self.command()
    
    def _on_enter(self, event):
        self.draw_button(hover=True)
    
    def _on_leave(self, event):
        self.draw_button(hover=False)

class RoundedFrame(tk.Frame):
    def __init__(self, parent, radius=25, bg="#ffffff", **kwargs):
        super().__init__(parent, **kwargs)
        self.radius = radius
        self.bg = bg
        self.canvas = tk.Canvas(self, bg=parent.cget('bg'), highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
    def draw_rounded_rect(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width > 1 and height > 1:
            points = [self.radius, 0,
                     width-self.radius, 0,
                     width, 0,
                     width, self.radius,
                     width, height-self.radius,
                     width, height,
                     width-self.radius, height,
                     self.radius, height,
                     0, height,
                     0, height-self.radius,
                     0, self.radius,
                     0, 0]
            self.canvas.create_polygon(points, smooth=True, fill=self.bg, outline="")

class VQ_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vector Quantization System")
        self.root.state("zoomed")  # Fullscreen / maximize
        
        # Blue/Violet color palette
        self.bg_color = "#1e2a4a"  # Dark blue background
        self.primary_color = "#8a2be2"  # Blue violet
        self.secondary_color = "#4a6fc7"  # Royal blue
        self.accent_color = "#ff6b6b"  # Coral accent for reset
        self.text_color = "#e6e6ff"  # Light lavender text
        self.frame_bg = "#2d3b5e"  # Medium blue frames
        self.button_color = "#6a5acd"  # Slate blue for buttons
        self.placeholder_bg = "#3a4a6b"  # Dark blue for placeholders
        
        self.root.configure(bg=self.bg_color)
        
        # Store images
        self.original_img = None
        self.decompressed_img = None
        self.original_panel = None
        self.decompressed_panel = None
        self.current_image_path = None

        # Log capturing
        self.log_content = ""
        self.log_widget = None
        sys.stdout = self
        sys.stderr = self
        
        # Title
        header_frame = tk.Frame(root, bg=self.primary_color, height=60)
        header_frame.pack(fill="x", pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="Vector Quantization System", 
            font=("Arial", 20, "bold"), 
            fg="white", 
            bg=self.primary_color,
            pady=20
        )
        title_label.pack()

        # frame for file selection and buttons
        top_frame = tk.Frame(root, bg=self.bg_color, pady=15)
        top_frame.pack(fill="x", padx=20)

        # File selection frame
        file_frame = tk.Frame(top_frame, bg=self.bg_color)
        file_frame.pack(fill="x", pady=(0, 15))
        
        tk.Label(
            file_frame, 
            text="Image Path:", 
            font=("Arial", 12, "bold"), 
            bg=self.bg_color,
            fg=self.text_color
        ).pack(side="left")

        self.path_entry = tk.Entry(
            file_frame, 
            width=60, 
            font=("Arial", 12),
            relief="flat",
            bd=2,
            bg="#f0f0f5",
            fg="#333333"
        )
        self.path_entry.pack(side="left", padx=10)

        # Use rounded button for browse
        RoundedButton(
            file_frame, 
            text="Browse", 
            command=self.browse, 
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            width=100,
            height=35
        ).pack(side="left")

        # Buttons row
        button_frame = tk.Frame(top_frame, bg=self.bg_color)
        button_frame.pack(fill="x", pady=10)

        # Create rounded buttons
        RoundedButton(
            button_frame, 
            text="Compress", 
            command=self.open_compress_window,
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            width=120,
            height=40
        ).pack(side="left", padx=8)

        RoundedButton(
            button_frame, 
            text="Decompress", 
            command=self.open_decompress_window,  # Changed to open popup
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            width=120,
            height=40
        ).pack(side="left", padx=8)

        RoundedButton(
            button_frame, 
            text="Show Codebook / Binary", 
            command=self.show_files,
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            width=160,
            height=40
        ).pack(side="left", padx=8)

        RoundedButton(
            button_frame, 
            text="Reset", 
            command=self.reset_images,
            bg=self.accent_color,
            fg="white",
            font=("Arial", 10, "bold"),
            width=100,
            height=40
        ).pack(side="left", padx=8)

        
        # Image Display Frames
        main_frame = tk.Frame(root, bg=self.bg_color)
        main_frame.pack(fill="both", expand=True, pady=10, padx=20)

        # Left frame → Original image
        self.original_frame = tk.LabelFrame(
            main_frame, 
            text="Original Image", 
            font=("Arial", 14, "bold"), 
            padx=10, 
            pady=10,
            bg=self.frame_bg,
            fg=self.text_color,
            relief="flat",
            bd=0
        )
        self.original_frame.pack(side="left", expand=True, fill="both", padx=10)

        # Right frame → Decompressed image
        self.decompressed_frame = tk.LabelFrame(
            main_frame, 
            text="Decompressed Image", 
            font=("Arial", 14, "bold"), 
            padx=10, 
            pady=10,
            bg=self.frame_bg,
            fg=self.text_color,
            relief="flat",
            bd=0
        )
        self.decompressed_frame.pack(side="right", expand=True, fill="both", padx=10)

        # Create rounded placeholder frames
        self.original_placeholder = self.create_rounded_placeholder(self.original_frame, "No image loaded")
        self.decompressed_placeholder = self.create_rounded_placeholder(self.decompressed_frame, "Decompressed image will appear here")

        print("Welcome to Vector Quantization System.")
        print("Ready for operations...\n")

    def write(self, text):
        self.log_content += text
        sys.__stdout__.write(text)
        if self.log_widget:
            try:
                self.log_widget.configure(state="normal")
                self.log_widget.insert("end", text)
                self.log_widget.see("end")
                self.log_widget.configure(state="disabled")
            except:
                pass

    def flush(self):
        pass

    def create_rounded_placeholder(self, parent, text):
        # Create a canvas for rounded placeholder
        canvas = tk.Canvas(parent, bg=self.frame_bg, highlightthickness=0, width=400, height=400)
        canvas.pack(expand=True)
        
        # Draw rounded rectangle
        self.draw_rounded_rect(canvas, 400, 400, 25, self.placeholder_bg)
        
        # Add text
        canvas.create_text(200, 200, text=text, fill=self.text_color, font=("Arial", 12), width=350)
        
        return canvas

    def draw_rounded_rect(self, canvas, width, height, radius, color):
        # Create points for rounded rectangle
        points = [radius, 0,
                 width-radius, 0,
                 width, 0,
                 width, radius,
                 width, height-radius,
                 width, height,
                 width-radius, height,
                 radius, height,
                 0, height,
                 0, height-radius,
                 0, radius,
                 0, 0]
        canvas.create_polygon(points, smooth=True, fill=color, outline="")

    
    # File browsing
    def browse(self):
        file = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, file)
            self.current_image_path = file
            self.show_original_image(file)

    def show_styled_error(self, title, message):
        """Show an error message with custom styling that matches our design"""
        error_popup = Toplevel(self.root)
        error_popup.title(title)
        error_popup.geometry("400x200")
        error_popup.configure(bg=self.bg_color)
        error_popup.resizable(False, False)
        
        # Center the popup
        error_popup.transient(self.root)
        error_popup.grab_set()
        
        # Position in center of parent window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (400 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (150 // 2)
        error_popup.geometry(f"+{x}+{y}")

        # Error icon and message
        tk.Label(
            error_popup, 
            text="⚠️",  # Error icon
            font=("Arial", 24),
            bg=self.bg_color,
            fg=self.accent_color
        ).pack(pady=(20, 10))

        tk.Label(
            error_popup, 
            text=message, 
            font=("Arial", 12),
            bg=self.bg_color,
            fg=self.text_color,
            wraplength=350
        ).pack(pady=(0, 20))

        # OK button 
        RoundedButton(
            error_popup, 
            text="OK", 
            command=error_popup.destroy,
            bg=self.accent_color,
            fg="white",
            font=("Arial", 10, "bold"),
            width=100,
            height=35
        ).pack(pady=10)

    def show_original_image(self, path):
        img = Image.open(path)
        
        # Set maximum dimensions (you can adjust these)
        max_width = 500
        max_height = 500
        
        # Calculate scaling factor while maintaining aspect ratio
        img_ratio = img.width / img.height
        
        if img.width > max_width or img.height > max_height:
            if img_ratio > 1:
                # Landscape image
                new_width = max_width
                new_height = int(max_width / img_ratio)
            else:
                # Portrait image
                new_height = max_height
                new_width = int(max_height * img_ratio)
            
            # Resize with high-quality filter
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create rounded image
        rounded_img = self.create_rounded_image(img, 25)
        self.original_img = ImageTk.PhotoImage(rounded_img)

        if self.original_panel:
            self.original_panel.config(image=self.original_img)
        else:
            self.original_panel = tk.Label(self.original_frame, image=self.original_img, bg=self.frame_bg)
            self.original_panel.pack(expand=True)
            
        # Remove placeholder if visible
        if self.original_placeholder.winfo_viewable():
            self.original_placeholder.pack_forget()

    def show_decompressed_image(self, path):
        img = Image.open(path)
        
        # Set maximum dimensions (you can adjust these)
        max_width = 500
        max_height = 500
        
        # Calculate scaling factor while maintaining aspect ratio
        img_ratio = img.width / img.height
        
        if img.width > max_width or img.height > max_height:
            if img_ratio > 1:
                # Landscape image
                new_width = max_width
                new_height = int(max_width / img_ratio)
            else:
                # Portrait image
                new_height = max_height
                new_width = int(max_height * img_ratio)
            
            # Resize with high-quality filter
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create rounded image
        rounded_img = self.create_rounded_image(img, 25)
        self.decompressed_img = ImageTk.PhotoImage(rounded_img)

        if self.decompressed_panel:
            self.decompressed_panel.config(image=self.decompressed_img)
        else:
            self.decompressed_panel = tk.Label(self.decompressed_frame, image=self.decompressed_img, bg=self.frame_bg)
            self.decompressed_panel.pack(expand=True)
                
        # Remove placeholder if visible
        if self.decompressed_placeholder.winfo_viewable():
            self.decompressed_placeholder.pack_forget()

    def create_rounded_image(self, img, radius):
        """Create an image with rounded corners"""
        mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # Draw rounded rectangle on mask
        width, height = img.size
        mask_draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)
        
        # Apply mask
        result = Image.new('RGBA', img.size, (0, 0, 0, 0))
        result.paste(img, mask=mask)
        return result

    # Compress popup window
    def open_compress_window(self):
        popup = Toplevel(self.root)
        popup.title("Compression Settings")
        popup.geometry("350x280")
        popup.configure(bg=self.bg_color)
        popup.resizable(False, False)
        
        # Center the popup
        popup.transient(self.root)
        popup.grab_set()
        
        # Position in center of parent window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (350 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (280 // 2)
        popup.geometry(f"+{x}+{y}")

        tk.Label(
            popup, 
            text="Compression Parameters", 
            font=("Arial", 14, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(pady=15)

        # Input fields
        input_frame = tk.Frame(popup, bg=self.bg_color)
        input_frame.pack(pady=15, padx=25, fill="x")
        
        tk.Label(
            input_frame, 
            text="Block Height:", 
            bg=self.bg_color,
            fg=self.text_color,
            font=("Arial", 10)
        ).grid(row=0, column=0, sticky="w", pady=8)
        bh_entry = tk.Entry(input_frame, width=15, relief="flat", bg="#f0f0f5")
        bh_entry.grid(row=0, column=1, sticky="e", pady=8)

        tk.Label(
            input_frame, 
            text="Block Width:", 
            bg=self.bg_color,
            fg=self.text_color,
            font=("Arial", 10)
        ).grid(row=1, column=0, sticky="w", pady=8)
        bw_entry = tk.Entry(input_frame, width=15, relief="flat", bg="#f0f0f5")
        bw_entry.grid(row=1, column=1, sticky="e", pady=8)

        tk.Label(
            input_frame, 
            text="Quantization Level (k):", 
            bg=self.bg_color,
            fg=self.text_color,
            font=("Arial", 10)
        ).grid(row=2, column=0, sticky="w", pady=8)
        k_entry = tk.Entry(input_frame, width=15, relief="flat", bg="#f0f0f5")
        k_entry.grid(row=2, column=1, sticky="e", pady=8)

        def confirm_settings():
            try:
                bh = int(bh_entry.get())
                bw = int(bw_entry.get())
                k = int(k_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integers for parameters.")
                return
            
            popup.destroy()
            self.run_compression_thread(bh, bw, k)

        button_frame = tk.Frame(popup, bg=self.bg_color)
        button_frame.pack(pady=15)
        
        RoundedButton(
            button_frame, 
            text="Start Compression", 
            command=confirm_settings,
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            width=140,
            height=35
        ).pack(side="left", padx=8)
        
        RoundedButton(
            button_frame, 
            text="Cancel", 
            command=popup.destroy,
            bg="#6c757d",
            fg="white",
            font=("Arial", 10),
            width=100,
            height=35
        ).pack(side="left", padx=8)

    def run_compression_thread(self, bh, bw, k):
        t = threading.Thread(target=self.run_compression, args=(bh, bw, k))
        t.daemon = True
        t.start()

    def run_compression(self, bh, bw, k):
        if not self.current_image_path:
            messagebox.showerror("Error", "No image selected.")
            return

        print("\n--- Starting Compression Process ---")
        try:
            cb = Codebook(self.current_image_path, bh, bw)
            cb.generate_codebook(k)
            cb.compress()
            print("--- Compression Completed Successfully ---")
            messagebox.showinfo("Success", "Compression completed! Files generated.")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    # NEW: Decompress popup window (matching design)
    def open_decompress_window(self):
        if not self.current_image_path:
            self.show_styled_error("Error", "Please select an original image first.")
            return

        popup = Toplevel(self.root)
        popup.title("Decompression Settings")
        popup.geometry("350x280")
        popup.configure(bg=self.bg_color)
        popup.resizable(False, False)
        
        # Center the popup
        popup.transient(self.root)
        popup.grab_set()
        
        # Position in center of parent window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (350 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (280 // 2)
        popup.geometry(f"+{x}+{y}")

        tk.Label(
            popup, 
            text="Decompression Parameters", 
            font=("Arial", 14, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(pady=15)

        # Input fields for decompression
        input_frame = tk.Frame(popup, bg=self.bg_color)
        input_frame.pack(pady=15, padx=25, fill="x")
        
        base = os.path.splitext(os.path.basename(self.current_image_path))[0]
        
        # Show which files will be loaded
        tk.Label(
            input_frame, 
            text="Files to load:", 
            bg=self.bg_color,
            fg=self.text_color,
            font=("Arial", 10, "bold")
        ).grid(row=0, column=0, sticky="w", pady=5)

        # File list with rounded backgrounds
        files_frame = tk.Frame(input_frame, bg=self.bg_color)
        files_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=10)
        
        file_names = [f"{base}_codebook.json", f"{base}_labels.json"]
        for i, file_name in enumerate(file_names):
            file_canvas = tk.Canvas(files_frame, bg=self.bg_color, highlightthickness=0, width=280, height=25)
            file_canvas.pack(pady=3)
            self.draw_rounded_rect(file_canvas, 280, 25, 12, self.placeholder_bg)
            file_canvas.create_text(140, 12, text=file_name, fill=self.text_color, font=("Arial", 9))

        def start_decompression():
            popup.destroy()
            self.run_decompression_thread(f"{base}_labels.json", f"{base}_codebook.json")

        button_frame = tk.Frame(popup, bg=self.bg_color)
        button_frame.pack(pady=15)
        
        RoundedButton(
            button_frame, 
            text="Start Decompression", 
            command=start_decompression,
            bg=self.button_color,
            fg="white",
            font=("Arial", 10, "bold"),
            width=150,
            height=35
        ).pack(side="left", padx=8)
        
        RoundedButton(
            button_frame, 
            text="Cancel", 
            command=popup.destroy,
            bg="#6c757d",
            fg="white",
            font=("Arial", 10),
            width=100,
            height=35
        ).pack(side="left", padx=8)

    def run_decompression_thread(self, labels_file, codebook_file):
        t = threading.Thread(target=self.run_decompression, args=(labels_file, codebook_file))
        t.daemon = True
        t.start()

    def run_decompression(self, labels_file, codebook_file):
        labels_path = os.path.join(script_dir, labels_file)
        codebook_path = os.path.join(script_dir, codebook_file)
        
        if not os.path.exists(labels_path) or not os.path.exists(codebook_path):
            messagebox.showerror("Error", f"Files not found:\n{labels_file}\n{codebook_file}")
            return

        print("\n--- Starting Decompression Process ---")
        base_name = os.path.splitext(os.path.basename(labels_path))[0].replace("_labels", "")
        output_path = os.path.join(script_dir, f"{base_name}_reconstructed_gui.png")

        try:
            Codebook.decompress(labels_path, codebook_path, output_path)
            print("--- Decompression Completed Successfully ---")
            
            # Update UI in main thread
            self.root.after(0, lambda: self.show_decompressed_image(output_path))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Decompression completed.\nSaved to: {output_path}"))
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred:\n{str(e)}"))

    # show generated files popup (Now shows Command Output)
    def show_files(self):
        popup = Toplevel(self.root)
        popup.title("Command Output / Logs")
        popup.geometry("600x400")
        popup.configure(bg=self.bg_color)
        
        # Center the popup
        popup.transient(self.root)
        
        # Position in center of parent window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (600 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (400 // 2)
        popup.geometry(f"+{x}+{y}")

        tk.Label(
            popup, 
            text="System Logs", 
            font=("Arial", 14, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(pady=10)

        # Log Text Area
        log_frame = tk.Frame(popup, bg=self.bg_color, padx=10)
        log_frame.pack(fill="both", expand=True)
        
        self.log_widget = scrolledtext.ScrolledText(
            log_frame, 
            state='normal', 
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            height=15
        )
        self.log_widget.pack(fill="both", expand=True)
        
        # Populate with existing logs
        self.log_widget.insert("end", self.log_content)
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

        def on_close():
            self.log_widget = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", on_close)

        RoundedButton(
            popup, 
            text="Close", 
            command=on_close,
            bg=self.button_color,
            fg="white",
            font=("Arial", 10),
            width=100,
            height=35
        ).pack(pady=10)

    # Reset images
    def reset_images(self):
        if self.original_panel:
            self.original_panel.pack_forget()
            self.original_panel = None
            
        if self.decompressed_panel:
            self.decompressed_panel.pack_forget()
            self.decompressed_panel = None

        # Clear the image path textbox
        self.path_entry.delete(0, tk.END)
    
        # Clear the current image path
        self.current_image_path = None

        self.original_img = None
        self.decompressed_img = None
        
        # Show placeholders again
        self.original_placeholder.pack(expand=True)
        self.decompressed_placeholder.pack(expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = VQ_GUI(root)
    root.mainloop()