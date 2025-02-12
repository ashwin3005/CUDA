import torch
import torch.nn.functional as F

vector = torch.arange(1, 21, dtype=torch.float32)
vector = vector.reshape(4, 5)


print("Input vector:")
print(vector)


# softmax along the last dimension
output = F.softmax(vector, dim=-1)
print("Output vector:")
print(output)