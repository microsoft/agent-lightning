import torch

print("Torch version:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("Built with ROCm:", torch.version.hip)

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("CUDA device capability:", torch.cuda.get_device_capability())
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print("TF32 supported:", torch.cuda.is_tf32_supported())

    print(f"{'GPU':<4} {'Name':<30} {'Memory Allocated/Reserved/Total'}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        allocated = torch.cuda.memory_allocated(i) // (1024 ** 2)
        reserved = torch.cuda.memory_reserved(i) // (1024 ** 2)
        total = torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)
        print(f"{i:<4} {name:<30} {allocated}MiB / {reserved}MiB / {total}MiB")

