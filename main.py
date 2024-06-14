import numpy as np
import matplotlib.pyplot as plt

# Convolution function
def apply_convolution(image, kernel, stride=1, padding='valid'):
    if padding == 'same':
        pad_h = (kernel.shape[0] - 1) // 2
        pad_w = (kernel.shape[1] - 1) // 2
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output_height = (image.shape[0] - kernel.shape[0]) // stride + 1
    output_width = (image.shape[1] - kernel.shape[1]) // stride + 1
    output = np.zeros((output_height, output_width))
    
    for y in range(0, image.shape[0] - kernel.shape[0] + 1, stride):
        for x in range(0, image.shape[1] - kernel.shape[1] + 1, stride):
            output[y // stride, x // stride] = np.sum(image[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel)
    
    return output

# Max pooling function
def max_pooling(image, size=2, stride=2):
    output_height = (image.shape[0] - size) // stride + 1
    output_width = (image.shape[1] - size) // stride + 1
    output = np.zeros((output_height, output_width))
    
    for y in range(0, image.shape[0] - size + 1, stride):
        for x in range(0, image.shape[1] - size + 1, stride):
            output[y // stride, x // stride] = np.max(image[y:y + size, x:x + size])
    
    return output

# Filters
vertical_edge_detection = np.array([[ 1,  0, -1],
                                    [ 1,  0, -1],
                                    [ 1,  0, -1]])

horizontal_edge_detection = np.array([[ 1,  1,  1],
                                      [ 0,  0,  0],
                                      [-1, -1, -1]])

edge_enhancement = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

gaussian_blur = (1/256) * np.array([[1,  4,  7,  4, 1],
                                    [4, 16, 26, 16, 4],
                                    [7, 26, 41, 26, 7],
                                    [4, 16, 26, 16, 4],
                                    [1,  4,  7,  4, 1]])

edge_enhancement_large = np.array([[-1, -1, -1, -1, -1],
                                   [-1,  2,  2,  2, -1],
                                   [-1,  2,  8,  2, -1],
                                   [-1,  2,  2,  2, -1],
                                   [-1, -1, -1, -1, -1]])

def plot_images(images, titles, cmap='gray', fig_title=None):
    n = len(images)
    plt.figure(figsize=(20, 5))
    if fig_title:
        plt.suptitle(fig_title, fontsize=16)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Load the image and convert to grayscale
image_path = 'yann_lecun.jpg'  # Replace with your image path
image = plt.imread(image_path)

# Convert to grayscale if the image is RGB
if image.ndim == 3 and image.shape[2] == 3:
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Apply filters
vertical_edges = apply_convolution(image, vertical_edge_detection, padding='same')
horizontal_edges = apply_convolution(image, horizontal_edge_detection, padding='same')
enhanced_edges = apply_convolution(image, edge_enhancement, padding='same')
blurred_image = apply_convolution(image, gaussian_blur, padding='same')
enhanced_edges_large = apply_convolution(image, edge_enhancement_large, padding='same')

# Display results in separate figures using subplot
plot_images([vertical_edges, horizontal_edges, enhanced_edges],
            ['Vertical Edge Detection', 'Horizontal Edge Detection', 'Edge Enhancement'],
            fig_title='Edge Detection and Enhancement')

plot_images([blurred_image, enhanced_edges_large],
            ['Gaussian Blur', 'Edge Enhancement Large'],
            fig_title='Blur and Large Edge Enhancement')

# Stack the feature maps into a 3D feature map
feature_map = np.stack([vertical_edges, horizontal_edges, enhanced_edges, blurred_image, enhanced_edges_large], axis=-1)

# Print shape of the feature map
print("Shape of the 3D feature map:", feature_map.shape)

# Apply max pooling on the feature map
pooled_feature_map = np.stack([max_pooling(vertical_edges), 
                               max_pooling(horizontal_edges), 
                               max_pooling(enhanced_edges), 
                               max_pooling(blurred_image), 
                               max_pooling(enhanced_edges_large)], axis=-1)

# Display pooled feature maps in a single figure using subplot
plot_images([pooled_feature_map[..., 0], pooled_feature_map[..., 1], pooled_feature_map[..., 2], 
             pooled_feature_map[..., 3], pooled_feature_map[..., 4]],
            ['Pooled Vertical Edge Detection', 'Pooled Horizontal Edge Detection', 'Pooled Edge Enhancement', 
             'Pooled Gaussian Blur', 'Pooled Edge Enhancement Large'],
            fig_title='Pooled Feature Maps')
