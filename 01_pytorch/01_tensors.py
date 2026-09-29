# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

import torch
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import cv2
import time

print("PyTorch version:" , torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. Using CPU.")

# Download some digit images from MNIST dataset

urlretrieve(
    "https://learnopencv.com/wp-content/uploads/2024/07/mnist_0.jpg",
    "mnist_0.jpg"
)

urlretrieve(
    "https://learnopencv.com/wp-content/uploads/2024/07/mnist_1.jpg",
    "mnist_1.jpg"
)


# Original image
digit_0_array_org = cv2.imread("mnist_0.jpg")
digit_1_array_org = cv2.imread("mnist_1.jpg")


# Grayscale image
digit_0_array_gray = cv2.imread(
    "mnist_0.jpg",
    cv2.IMREAD_GRAYSCALE
)
digit_1_array_gray = cv2.imread(
    "mnist_1.jpg",
    cv2.IMREAD_GRAYSCALE
)

print("Original image shape:", digit_0_array_org.shape)
print("Grayscale image shape:", digit_0_array_gray.shape)

fig,axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(digit_0_array_org)
axs[0].set_title("Original Image")
axs[0].axis('off')
axs[1].imshow(digit_0_array_gray, cmap='gray')
axs[1].set_title("Grayscale Image")
axs[1].axis('off')
plt.show(block=False)
plt.pause(5)
plt.close()

#print("Original image array", digit_1_array_gray)

#convert original images from numpy array to torch tensor   and normalize the pixel values to [0, 1]
digit_0_tensor = torch.tensor(digit_0_array_org, dtype=torch.float32) / 255.0
digit_1_tensor = torch.tensor(digit_1_array_org, dtype=torch.float32) / 255.0

print("Digit 0 tensor shape:", digit_0_tensor.shape)
print("Digit 1 tensor shape:", digit_1_tensor.shape)
print("min and max values of digit 0 tensor:", digit_0_tensor.min().item(), digit_0_tensor.max().item())
print("min and max values of digit 1 tensor:", digit_1_tensor.min().item(), digit_1_tensor.max().item())

plt.imshow(digit_0_tensor, cmap='gray')
plt.axis('off')
plt.show(block=False)
plt.pause(5)
#After showing images are blocking the code execution, so we will use plt.close() to close the figure after displaying it
plt.close()

#creating input tensor for the model by stacking the two digit tensors along a new dimension
batch_tensor = torch.stack([digit_0_tensor, digit_1_tensor], dim=0)

print("Input tensor shape:", batch_tensor.shape)

batch_input = batch_tensor.permute(0, 3, 1, 2)  # Change shape to (batch_size, channels, height, width)
print("Batch input shape:", batch_input.shape)


# CPU vs GPU computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    batch_input = batch_input.to(device)
    print("Batch input moved to GPU.")
    start_time = time.time()
    # Simulate some computation
    result = batch_input * 0.1
    torch.cuda.synchronize()  # Wait for all GPU operations to finish
    end_time = time.time()
    print("GPU processing time:", end_time - start_time, "seconds")

#CPU
device = torch.device("cpu")
batch_input = batch_input.to(device)
print("Batch input moved to CPU.")
start_time = time.time()
# Simulate some computation
result = batch_input * 0.1
end_time = time.time()
print("CPU processing time:", end_time - start_time, "seconds")


import time
import torch

# Assume batch_input is already created.

if torch.cuda.is_available():
    gpu_input = batch_input.to("cuda")

    # Warm up GPU
    for _ in range(10):
        result = gpu_input * 0.1

    torch.cuda.synchronize()

    start_time = time.perf_counter()

    result = gpu_input * 0.1

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    print("GPU processing time:",
          end_time - start_time, "seconds")

# CPU benchmark
cpu_input = batch_input.to("cpu")

# Warm up CPU
for _ in range(10):
    result = cpu_input * 0.1

start_time = time.perf_counter()

result = cpu_input * 0.1

end_time = time.perf_counter()

print("CPU processing time:",
      end_time - start_time, "seconds")
