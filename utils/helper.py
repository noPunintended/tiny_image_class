def convert_to_RGB(image):
    """Convert grayscale image to RGB by duplicating channels."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image