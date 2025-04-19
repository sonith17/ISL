from torch.utils.data import DataLoader,Dataset,Subset, random_split
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
import torchvision
from tqdm import tqdm

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

train_size = int(0.77 * len(filtered_dataset))
test_size = len(filtered_dataset) - train_size
train_data, test_data = random_split(filtered_dataset, [train_size, test_size])

train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(test_data,batch_size=32,shuffle=True)

x,y = train_data[0]
print(x.shape,y)

print(f"training samples: {len(train_data)}")
print(f"test samples: {len(test_data)}")
print(f"Classes {len(selected_classes)}")

class SimpleNN(torch.nn.Module):
  def __init__(self):
    super(SimpleNN,self).__init__()
    self.conv = torch.nn.Conv2d(3,32,2,padding=1,stride=2)
    self.conv2 = torch.nn.Conv2d(32,64,2,padding=1,stride=2)
    self.conv3 = torch.nn.Conv2d(64,128,2,padding=1,stride=2)
    self.conv4 = torch.nn.Conv2d(128,256,2,padding=1,stride=2)
    self.conv5 = torch.nn.Conv2d(256,256,2,padding=1,stride=2)
    self.pool = torch.nn.MaxPool2d(2)
    self.flatten = torch.nn.Flatten()
    self.relu = torch.nn.ReLU()
    self.fc1 = torch.nn.Linear(16384, len(selected_classes))

  def forward(self,x):
    x = self.relu(self.conv(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    x = self.flatten(self.conv5(x))
    x = self.fc1(x)
    return x

model = SimpleNN()
epochs = 10
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
correct = 0
total = 0

model.to(device)

model.train()
for epoch in range(epochs):
  for i,(input,label) in enumerate(tqdm(train_loader,desc=f"Training {epoch}/{epochs} ")):
    input,label = input.to(device),label.to(device)
    output = model(input)
    loss = criterion(output,label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    _,predicted = torch.max(output.data,1)
    total += label.size(0)
    correct += (predicted == label).sum().item()

print(f'Loss: {loss.item()}')

print(f'Train Accuracy: {(correct/total)*100}%')

correct = 0
total = 0
model.eval()
for i,(input,label) in enumerate(test_loader):
  input,label = input.to(device),label.to(device)
  output = model(input)
  _,predicted = torch.max(output.data,1)
  total += label.size(0)
  correct += (predicted == label).sum().item()

print(f'Test Accuracy: {(correct/total)*100}%')




