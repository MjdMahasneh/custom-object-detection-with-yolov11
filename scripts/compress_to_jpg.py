import os
from PIL import Image

def convert_png_to_jpeg(input_folder, quality=85):
    """
    Convert PNG images to JPEG with compression.

    :param input_folder: Folder containing PNG images.
    :param quality: JPEG compression quality (0-100), higher means better quality.
    """
    # Create 'compressed' folder if it doesn't exist
    compressed_folder = os.path.join(input_folder, 'compressed')
    os.makedirs(compressed_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(compressed_folder, os.path.splitext(filename)[0] + '.jpg')

            try:
                img = Image.open(input_path).convert("RGB")  # Convert to RGB for JPEG
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                print(f"Converted & Compressed: {filename} -> {output_path}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Example usage
input_folder = 'dataset'  # Replace with your folder path
convert_png_to_jpeg(input_folder, quality=85)  # Adjust quality (0-100)
