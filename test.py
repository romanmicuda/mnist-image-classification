import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 7 * 7 * 128)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def main():
    parser = argparse.ArgumentParser(description="A simple program to recognize digits")
    parser.add_argument('--model_path', type=str, help="Enter input model path")
    parser.add_argument('--digit_path', type=str, help="Enter input image path")
    args = parser.parse_args()
    test(args.model_path, args.digit_path)

def test(model_path, image_path):
    print("hee")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert('L')

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = SimpleCNN().to(device)  # Create model instance and move to device
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        print(f'Predicted digit: {predicted.item()}')

main()