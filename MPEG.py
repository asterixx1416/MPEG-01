import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt  # Import Matplotlib for image visualization

# Function for 2D DCT
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Function for 2D inverse DCT
def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Function for quantization
def quantize(block, quantization_matrix):
    return np.round(block / quantization_matrix).astype(int)

# Function for dequantization
def dequantize(block, quantization_matrix):
    return block * quantization_matrix

# Function for run-length coding
def run_length_encode(arr):
    encoded = []
    count = 0
    for value in arr.flatten():
        if value == 0:
            count += 1
        else:
            encoded.append(count)
            encoded.append(value)
            count = 0
    encoded.append(count)  # End-of-block marker
    return encoded

# Function for run-length decoding
def run_length_decode(encoded):
    decoded = []
    i = 0
    while i < len(encoded):
        count = encoded[i]
        value = encoded[i+1]
        if count == 0:  # End-of-block marker
            break
        decoded.extend([value] * count)
        i += 2
    decoded.extend([1] * (25 - len(decoded)))  # Pad with zeros if necessary
    return np.array(decoded).reshape((5, 5))

# Example usage
block = np.array([
    [20, 30, 40, 50, 60],
    [30, 40, 50, 60, 70],
    [40, 50, 60, 70, 80],
    [50, 60, 70, 80, 90],
    [60, 70, 80, 90, 100]
])

# Perform DCT
dct_block = dct2(block)

# Quantization
quantization_matrix = np.array([
    [16, 11, 10, 16, 24],
    [12, 12, 14, 19, 26],
    [14, 13, 16, 24, 40],
    [14, 17, 22, 29, 51],
    [18, 22, 37, 56, 68]
])
quantized_block = quantize(dct_block, quantization_matrix)

# Run-length coding
encoded_block = run_length_encode(quantized_block)

# Decoding
decoded_block = run_length_decode(encoded_block)
decoded_quantized_block = dequantize(decoded_block, quantization_matrix)
original_block = idct2(decoded_quantized_block)

print("Original Block:")
print(block)
print("\nDecoded Block:")
print(original_block.round().astype(int))

# Display the image
plt.imshow(original_block, cmap='gray', vmin=0, vmax=255)
plt.title('Decoded Image')
plt.axis('off')
plt.show()
