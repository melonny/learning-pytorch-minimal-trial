from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize(32),
    transforms.ToTensor()
])
data_dir = "F:\\DogEmotion"
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(len(full_dataset) * 0.8)
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, num_workers=0, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, num_workers=0, shuffle=True)