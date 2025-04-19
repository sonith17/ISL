from torch.utils.data import DataLoader,Dataset,Subset, random_split
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
import torchvision
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

selected_classes = [
    "abacus", "abaya", "academic_gown", "accordion", "acorn",
    "acorn_squash", "acoustic_guitar", "admiral", "affenpinscher", "afghan_hound",
    "african_chameleon", "african_crocodile", "african_elephant", "african_grey", "african_hunting_dog",
    "agama", "agaric", "aircraft_carrier", "airedale", "airliner", "airship"
]

data = datasets.ImageFolder(root='/scratch/data/imagenet-256/versions/1',transform=transform)

class_to_idx = data.class_to_idx
#print(class_to_idx)
selected_indices = [class_to_idx[cls] for cls in selected_classes]

filtered_indices = [
    idx for idx, (path, label) in enumerate(data.samples) if label in selected_indices
]

filtered_dataset = Subset(data, filtered_indices)


data_loader = DataLoader(filtered_dataset,batch_size=32,shuffle=True)


print(f"total samples: {len(filtered_dataset)}")




model = torch.jit.load('/scratch/isl_39/Lab_4_2_model.pt')
print("model loaded")

correct = 0
total = 0
model.eval()
for i,(input,label) in enumerate(tqdm(data_loader,desc="Testing")):
  input,label = input.to(device),label.to(device)
  output = model(input)
  _,predicted = torch.max(output.data,1)
  total += label.size(0)
  correct += (predicted == label).sum().item()

print(f'Test Accuracy: {(correct/total)*100}%')




