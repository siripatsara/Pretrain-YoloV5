import torch
from ultralytics import settings
# print(torch.__version__)
# print(torch.cuda.is_available())


# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]


def main():
    print("Hello from program!")

    # ตรวจสอบว่ามี CUDA หรือไม่
    print(f"CUDA available: {torch.cuda.is_available()}")

    # จำนวน GPU ทั้งหมด
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    # ข้อมูลแต่ละ GPU
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
