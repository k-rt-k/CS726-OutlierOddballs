import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse

# Use ResNet from torchvision
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()  # Remove the final fully connected layer

# Data transformation for evaluation
eval_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def main(data_path, output_path, args):
    # Load CIFAR-10 dataset
    dataroot = data_path # Change this path
    print(f"[My-Log][dataroot]{dataroot}")

    dataset = datasets.CIFAR100(root=dataroot, train=not args.test, download=True, transform=eval_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    embeddings, labels = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Model embedding generation")
    for images, targets in tqdm(dataloader):
        with torch.no_grad():
            images = images.to(device)
            output = model(images)
            embeddings.append(output.detach().cpu().numpy())
            labels.append(targets.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels).squeeze()

    data = {'embeddings': embeddings, 'labels': labels}

    data_path = os.path.join(output_path, "cifar100_{}_embeddings.pkl".format("train" if not args.test else "test"))
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Data saved to: {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embeddings generator')
    parser.add_argument('--data_path', type=str, help='The path of the data whose embeddings are to be generated')
    parser.add_argument('--output_path', type=str, default='./embeddings/', help='Path of the folder where .pkl files of embeddings are to be saved')
    parser.add_argument('--test', action='store_true', help='True for embedddings of test set, False for embeddings of train set')
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)
    main(data_path, output_path, args)
