import os
import cv2

def convert_to_png(input_folder):
    supported_formats = ('.avif', '.webp', '.jpg')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(input_folder, os.path.splitext(filename)[0] + '.png')

            try:
                img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("Image could not be read.")
                cv2.imwrite(output_path, img)
                print(f"Converted: {filename} -> {output_path}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Example usage
input_folder = 'dataset'  # Replace with your folder path
convert_to_png(input_folder)
