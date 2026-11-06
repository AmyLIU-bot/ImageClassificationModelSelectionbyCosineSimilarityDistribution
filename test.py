import torch

# 1. Check if CUDA (NVIDIA GPU support) is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # 2. Get the number of available GPUs
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    # 3. Get the name of the current GPU
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 4. Perform a quick tensor test on the GPU
    x = torch.rand(3, 3).cuda()
    print("Test Tensor on GPU successfully:", x.device)
else:
    print("PyTorch cannot see your GPU. It will default to CPU.")

