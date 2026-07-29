from pathlib import Path
import numpy as np



#- Generated image datasets -#

# Apply curve to normalized gradient
def apply_curve(x, curve):
    if curve == 'linear':
        return x
    elif curve == 'quadratic':
        return x ** 2
    elif curve == 'sqrt':
        return np.sqrt(x)
    elif curve == 'log':
        return np.log1p(x * 9) / np.log(10)  # scale [0,1] into [0,1] log space
    elif curve == 'exp':
        return (np.exp(x * 3) - 1) / (np.exp(3) - 1)  # normalized exponential
    elif curve == 'sigmoid':
        return 1 / (1 + np.exp(-10 * (x - 0.5)))  # smooth S-curve
    elif curve == 'sin':
        return 0.5 * (1 - np.cos(np.pi * x))  # smooth start and end
    else:
        raise ValueError(f"Unknown curve type: {curve}")
    

# Get image shape from number of pixels
def get_image_shape(total_pixels):
    for h in range(int(total_pixels ** 0.5), 0, -1):
        if total_pixels % h == 0:
            return h, total_pixels // h
    
    raise ValueError(f"Could not find image shape for {total_pixels} pixels.")


# Build my own dataset of images: gradient images
def create_gradients(total_pixels, directions=None, curves=None, height=None, width=None):
    if directions is None:
        directions = [
            'top_left_to_bottom_right'
        ]
    if curves is None:
        curves = ['linear', 'quadratic', 'sqrt', 'log', 'exp', 'sigmoid', 'sin']

    if height is None or width is None:
        height, width = get_image_shape(total_pixels)

    gradients = []
    i, j = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Precompute normalized coordinate matrices for all directions
    norm_maps = {
        'left_to_right': np.tile(np.linspace(0, 1, width), (height, 1)),
        'right_to_left': np.tile(np.linspace(1, 0, width), (height, 1)),
        'top_to_bottom': np.tile(np.linspace(0, 1, height)[:, np.newaxis], (1, width)),
        'bottom_to_top': np.tile(np.linspace(1, 0, height)[:, np.newaxis], (1, width)),
        'top_left_to_bottom_right': (i + j) / (width + height - 2),
        'bottom_right_to_top_left': ((height - 1 - i) + (width - 1 - j)) / (width + height - 2),
        'top_right_to_bottom_left': (i + (width - 1 - j)) / (width + height - 2),
        'bottom_left_to_top_right': ((height - 1 - i) + j) / (width + height - 2)
    }

    for direction in directions:
        if direction not in norm_maps:
            raise ValueError(f"Unknown direction: {direction}")
        base_map = norm_maps[direction]

        for curve in curves:
            curved_map = apply_curve(base_map, curve)
            gradients.append(curved_map)

    image_array = np.array(gradients).reshape(-1, height, width)
    return image_array


# Create dataset depending on source type
def create_images_dataset(source, parameters):
    if source == 'generated_gradients':
        X = create_gradients(**parameters)
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    return X



#- Dataset visualization -#

# Show image dataset
def show_images_dataset(X):
    import matplotlib.pyplot as plt

    for i in range(len(X)):
        image = X[i]
        if hasattr(image, 'detach'):
            image = image.detach().cpu()

        plt.subplot(1, len(X)+1, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()
    print('dataset shape:', X.shape, '\ndata type:', X.dtype)



#- Dataset save -#

QGAN_ROOT = Path(__file__).resolve().parents[2]
DATASETS_PATH = QGAN_ROOT / 'datasets'
DATASET_FILE_EXTENSION = '.npz'


# Get dataset file path
def get_dataset_filename(config):
    dataset_id = config['dataset']['id']

    return DATASETS_PATH / 'prepared' / f"{dataset_id}{DATASET_FILE_EXTENSION}"


# Create dataset file
def create_images_dataset_file(source, parameters, filename):
    X = create_images_dataset(source, parameters)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filename, X=X)
    print("Dataset file created.")


# Load dataset file
def load_images_dataset_file(filename):
    with np.load(filename) as data:
        X = data['X']

    return X



#- Dataset management -#

# Get dataset
def get_images_dataset(config):
    dataset_options = config['dataset']

    filename = get_dataset_filename(config)

    if dataset_options['reset'] or not filename.exists():
        create_images_dataset_file(dataset_options['source'], 
                            dataset_options['parameters'], 
                            filename
                            )

    return load_images_dataset_file(filename)
