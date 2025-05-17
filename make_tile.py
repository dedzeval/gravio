from PIL import Image
import numpy as np

def make_tileable(image_path, output_path):
    """
    Convert a source image into a smooth, tileable image.

    Args:
        image_path (str): Path to the source image.
        output_path (str): Path to save the tileable image.
    """
    # Load the source image
    img = Image.open(image_path)
    W, H = img.size  # Width and height of the image

    # Convert the image to a NumPy array for processing
    original = np.array(img, dtype=np.float32)

    # Define the blend width (e.g., 1/4 of the smaller dimension)
    S = min(W, H) // 4

    # Step 1: Create a horizontally tileable image
    hori = np.zeros_like(original, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            if x < S:
                # Left edge: blend with pixels from the right
                t = x / S  # Gradient from 0 (left) to 1 (right)
                hori[y, x] = (1 - t) * original[y, x] + t * original[y, (x - S) % W]
            elif x >= W - S:
                # Right edge: blend with pixels from the left
                t = (W - x) / S  # Gradient from 1 (right) to 0 (left)
                hori[y, x] = (1 - t) * original[y, x] + t * original[y, (x + S) % W]
            else:
                # Middle: keep original pixels
                hori[y, x] = original[y, x]

    # Step 2: Create a fully tileable image by blending vertically
    final = np.zeros_like(hori, dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if y < S:
                # Top edge: blend with pixels from the bottom
                t = y / S  # Gradient from 0 (top) to 1 (bottom)
                final[y, x] = (1 - t) * hori[y, x] + t * hori[(y - S) % H, x]
            elif y >= H - S:
                # Bottom edge: blend with pixels from the top
                t = (H - y) / S  # Gradient from 1 (bottom) to 0 (top)
                final[y, x] = (1 - t) * hori[y, x] + t * hori[(y + S) % H, x]
            else:
                # Middle: keep horizontally blended pixels
                final[y, x] = hori[y, x]

    # Clip values to [0, 255] and convert to uint8
    final = np.clip(final, 0, 255).astype(np.uint8)

    # Convert back to an image and save
    result = Image.fromarray(final)
    result.save(output_path)
    print(f"Tileable image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    source_image = "assets/galaxy_bg_1.png"   # Replace with your source image path
    output_image = "assets/tile-galaxy_bg_1.png"  # Output path
    make_tileable(source_image, output_image)