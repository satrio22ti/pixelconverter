import io
import cv2
import numpy as np
from flask import Flask, request, send_file, render_template

from rembg import remove, new_session
print("Sedang memuat model AI ke RAM, mohon tunggu...")
session_ai = new_session("u2netp")
print("Model AI siap digunakan!")

app = Flask(__name__)

GAMEBOY_PALETTE = np.array([(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)], dtype=np.uint8)
GAMEBOY_POCKET_PALETTE = np.array([(44, 50, 34), (79, 89, 57), (140, 152, 96), (198, 209, 145)], dtype=np.uint8)
NES_PALETTE = np.array([(0, 0, 0), (252, 252, 252), (188, 188, 188), (124, 124, 124)], dtype=np.uint8)

PALETTES = {"none": None, "gameboy": GAMEBOY_PALETTE, "gameboy_pocket": GAMEBOY_POCKET_PALETTE, "nes": NES_PALETTE}

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def snap_to_palette(img_rgb, palette, mask=None):
    h, w, _ = img_rgb.shape
    out = img_rgb.copy().astype(np.float32)
    pal = palette.astype(np.float32)
    
    for y in range(h):
        for x in range(w):
            if mask is not None and not mask[y, x]:
                continue
            old_color = out[y, x]
            dists = np.sum((pal - old_color) ** 2, axis=1)
            best_idx = np.argmin(dists)
            new_color = pal[best_idx]
            out[y, x] = new_color
            
            err = old_color - new_color
            if x + 1 < w: out[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0: out[y + 1, x - 1] += err * 3 / 16
                out[y + 1, x]     += err * 5 / 16
                if x + 1 < w: out[y + 1, x + 1] += err * 1 / 16
                
    return np.clip(out, 0, 255).astype(np.uint8)

def pixel_size_to_block(img, pixel_size):
    h, w = img.shape[:2]
    return max(1, round(max(h, w) / pixel_size))

def pixelate_dominant_color(img, block_size):
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    out = np.zeros_like(img)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img[y:y + block_size, x:x + block_size]
            flat_block = block.reshape(-1, channels) if channels > 1 else block.reshape(-1, 1)
            colors, counts = np.unique(flat_block, axis=0, return_counts=True)
            out[y:y + block_size, x:x + block_size] = colors[np.argmax(counts)]
    return out

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_bg_endpoint():
    if 'image' not in request.files: return "No image", 400
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        result_rgba = np.array(remove(img_rgb, session=session_ai))
        
        result_out = cv2.cvtColor(result_rgba, cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        print(f"Rembg Gagal: {e}")
        return "Gagal memproses AI Hapus Background.", 500
        
    _, buffer = cv2.imencode(".png", result_out)
    return send_file(io.BytesIO(buffer), mimetype='image/png')

@app.route('/process', methods=['POST'])
def process_endpoint():
    if 'image' not in request.files: return "No image", 400
    file = request.files['image']
    pixel_size = int(request.form.get('pixel_size', 32))
    palette_choice = request.form.get('palette', 'none')

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img_input = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if img_input.shape[2] == 4:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGRA2RGBA)
    else:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

    palette = None
    color_mode = "none"
    if palette_choice == "custom":
        c1, c2 = request.form.get('c1', '#0f380f'), request.form.get('c2', '#306230')
        c3, c4 = request.form.get('c3', '#8bac0f'), request.form.get('c4', '#9bbc0f')
        palette = np.array([hex_to_rgb(c1), hex_to_rgb(c2), hex_to_rgb(c3), hex_to_rgb(c4)], dtype=np.uint8)
        color_mode = "palette"
    elif palette_choice in PALETTES and PALETTES[palette_choice] is not None:
        palette = PALETTES[palette_choice]
        color_mode = "palette"

    has_alpha = img_input.shape[2] == 4
    pixelated = pixelate_dominant_color(img_input, pixel_size_to_block(img_input, pixel_size))
    
    if has_alpha:
        rgb = pixelated[:, :, :3]
        alpha = pixelated[:, :, 3]
        opaque_mask = alpha > 10
        rgb_out = snap_to_palette(rgb, palette, mask=opaque_mask) if color_mode == "palette" else rgb
        result_out = cv2.cvtColor(np.dstack([rgb_out, alpha]), cv2.COLOR_RGBA2BGRA)
    else:
        result = snap_to_palette(pixelated, palette) if color_mode == "palette" else pixelated
        result_out = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode(".png", result_out)
    return send_file(io.BytesIO(buffer), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)