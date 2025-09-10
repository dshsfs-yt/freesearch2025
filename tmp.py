import torch

# 현재 GPU 번호 (0번 기준)
device = torch.device("cuda:0")

# 총 GPU 메모리 (bytes 단위 → GB로 변환)
total = torch.cuda.get_device_properties(device).total_memory / 1024**3

# 이미 PyTorch에 의해 예약된 메모리
reserved = torch.cuda.memory_reserved(device) / 1024**3

# 실제 사용 중인 메모리
allocated = torch.cuda.memory_allocated(device) / 1024**3

# 사용 가능 추정치 (PyTorch 입장에서)
free_inside_reserved = reserved - allocated
free_total_estimate = total - reserved

print(f"Total GPU Memory: {total:.2f} GB")
print(f"Reserved by PyTorch: {reserved:.2f} GB")
print(f"Allocated by PyTorch: {allocated:.2f} GB")
print(f"Free inside reserved: {free_inside_reserved:.2f} GB")
print(f"Estimated Available for New Allocations: {free_total_estimate:.2f} GB")
