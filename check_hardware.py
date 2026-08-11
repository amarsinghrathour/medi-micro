import torch

def check_hardware():
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if MPS is built and available
    is_built = torch.backends.mps.is_built()
    is_available = torch.backends.mps.is_available()
    
    print(f"MPS Built: {is_built}")
    print(f"MPS Available: {is_available}")
    
    device = torch.device("mps" if is_available else "cpu")
    print(f"Using Device: {device}")
    
    if str(device) == "mps":
        print("✅ Success: Apple Silicon (MPS) is fully supported and enabled for PyTorch.")
        # Perform a quick tensor operation on MPS to verify
        x = torch.ones(1, device=device)
        print(f"Tensor on {device}: {x}")
    else:
        print("❌ Warning: PyTorch is not using MPS. Training will be extremely slow on CPU.")

if __name__ == "__main__":
    check_hardware()
