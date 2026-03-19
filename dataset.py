import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

class CatDogDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = 0 if 'Cat' in img_path else 1 
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            return self.__getitem__(random.randint(0, len(self.file_paths) - 1))

        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_dataloaders(data_dir, batch_size=128):
    cat_dir = os.path.join(data_dir, 'Cat')
    dog_dir = os.path.join(data_dir, 'Dog')
    
    all_files = []
    for d in [cat_dir, dog_dir]:
        for file in os.listdir(d):
            if file.endswith('.jpg'):
                all_files.append(os.path.join(d, file))
                
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CatDogDataset(train_files, transform=train_transform)
    test_dataset = CatDogDataset(test_files, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, test_loader
