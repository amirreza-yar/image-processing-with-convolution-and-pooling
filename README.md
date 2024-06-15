# Image Processing with Convolution and Pooling

## Overview

This project focuses on implementing and applying 2D convolution and max pooling on images to extract useful features like edges, textures, and patterns. The main objective is to demonstrate the use of different convolution filters and visualize the results using Python, NumPy, and Matplotlib.

## Features

- Implementation of 2D convolution
- Application of convolution filters:
  - Vertical Edge Detection
  - Horizontal Edge Detection
  - Edge Enhancement
- Visualization of results
- Utilization of Python libraries: NumPy and Matplotlib

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:
- `numpy`
- `matplotlib`

You can install them using pip:
```bash
pip install numpy matplotlib scipy
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-processing-with-convolution-and-pooling.git
```

2. Navigate to the project directory:
```bash
cd image-processing-with-convolution-and-pooling
```

3. Run the Python script:
```bash
python main.py
```

### Applying Convolution Filters

The project applies the following convolution filters to an input image:

- **Vertical Edge Detection**
- **Horizontal Edge Detection**
- **Edge Enhancement**

Make sure to replace the image path in the script with your own image path.

```python
# Load the image
image = plt.imread('path_to_your_image.jpg') # Replace with your image path
```

### Example Results

Here are some example results of applying the convolution filters:

- Original Image
- Vertical Edge Detection
- Horizontal Edge Detection
- Edge Enhancement

## Project Structure

```
│
├── main.py          # Main script to run the project
├── README.md        # Project documentation
└── notebook/          # Directory to store notebook files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by the need to understand and implement basic image processing techniques.
- Thanks to the open-source community for providing valuable resources and libraries.
