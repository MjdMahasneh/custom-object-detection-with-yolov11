import os
import cv2


def compress_png_images(input_folder, compression_level=9):
    """
    Compress PNG images and save them into a 'compressed' subfolder.

    :param input_folder: Folder containing PNG images.
    :param compression_level: PNG compression level (0-9), higher means more compression.
    """
    # Create 'compressed' folder if it doesn't exist
    compressed_folder = os.path.join(input_folder, 'compressed')
    os.makedirs(compressed_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(compressed_folder, filename)

            try:
                img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("Image could not be read.")
                # Compress and save PNG
                cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
                print(f"Compressed: {filename} -> {output_path}")
            except Exception as e:
                print(f"Failed to compress {filename}: {e}")


# Example usage
input_folder = 'dataset'  # Replace with your folder path
compress_png_images(input_folder, compression_level=9)  # Compression level 0-9
