from PIL import Image

def extract_resolution_and_grayscale(image_path):
    # Open the image
    image = Image.open(image_path).convert(‘L’)  # Convert to grayscale
    width, height = image.size  # Extract resolution
    grayscale_image = image  # Grayscale image

    print(f »Resolution: {width}x{height} »)
    grayscale_image.save(« grayscale_image.jpg »)  # Save grayscale image
    return width, height, grayscale_image

# Example usage
image_path = « black_and_white_photo.jpg »
resolution_width, resolution_height, grayscale_image = extract_resolution_and_grayscale(image
_path)