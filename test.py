import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from models.net import AdaIFL


def load_model(model_path, device):
    model = AdaIFL().to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    return model

def preprocess_image(img_path, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0).to(device)
    return img

def save_mask(mask, save_path):
    mask = (mask > 0.5).float() * 255
    mask = mask.squeeze().cpu().detach().numpy()
    plt.imsave(save_path, mask, cmap='gray')

def test(model_path, img_path, folder_path, device='cuda'):
    model = load_model(model_path, device)
    img = preprocess_image(img_path, device)
    
    with torch.no_grad():
        mask_pred = model(img)
    
    mask_pred = torch.sigmoid(mask_pred)
    image_name = os.path.basename(img_path)
    save_mask(mask_pred, os.path.join(folder_path, image_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Forgery Localization")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='./AdaIFL_v0.pth', help='Path to the pre-trained AdaIFL model')
    parser.add_argument('--output', type=str, default='./results', help='Directory to save the result mask')
    args = parser.parse_args()
    test(args.model, args.image, args.output)