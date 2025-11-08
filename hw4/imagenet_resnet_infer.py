import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import json
import os 
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from skimage.transform import resize
from skimage.segmentation import slic
from sklearn.linear_model import Ridge

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]   
    )
])

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}

def lime(image, model, num_samples=100, num_features = 10):
    img_array = np.array(image)
    segments = slic(img_array, n_segments=50, compactness=10, sigma=1, start_label=0)
    num_superpixels = np.unique(segments).shape[0]
    input_tensor = preprocess(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        original_output = model(input_tensor)
        original_probs = torch.nn.functional.softmax(original_output, dim=1)
        predicted_class = torch.argmax(original_probs, dim=1).item()

    perturbations = np.random.binomial(1, 0.5, size=(num_samples, num_superpixels))
    predictions = []
    distances = []
    for perturbation in perturbations:
        perturbed_image = img_array.copy()
        for i, active in enumerate(perturbation):
            if active == 0:
                perturbed_image[segments == i] = [128, 128, 128]
        perturbed_pil = Image.fromarray(perturbed_image.astype("uint8"))
        perturbed_tensor = preprocess(perturbed_pil).unsqueeze(0)
        if torch.cuda.is_available():
            perturbed_tensor = perturbed_tensor.to('cuda')
        with torch.no_grad():
            output = model(perturbed_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            predictions.append(probs[0, predicted_class].cpu().item())
        distance = np.sum(perturbation) / num_superpixels
        distances.append(distance)

    predictions = np.array(predictions)
    distances = np.array(distances)
    kernel_width = 0.25
    weights_samples = np.exp(-(1 - distances) ** 2 / (kernel_width ** 2))
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(perturbations, predictions, sample_weight=weights_samples)
    weights = ridge_model.coef_
    top_features = np.argsort(np.abs(weights))[-num_features:]
    explanation_mask = np.zeros_like(segments, dtype=float)
    for feature_idx in top_features:
        explanation_mask[segments == feature_idx] = np.abs(weights[feature_idx])
    return explanation_mask, weights, segments

def smoothgrad_explain(image, model, num_samples=50, noise_level=0.15):
    input_tensor = preprocess(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    total_gradients = None
    
    for _ in range(num_samples):
        noisy_input = input_tensor.clone()
        noise = torch.randn_like(noisy_input) * noise_level
        noisy_input = noisy_input + noise
        noisy_input.requires_grad = True
        output = model(noisy_input)
        model.zero_grad()
        output[0, predicted_class].backward()
        gradients = noisy_input.grad.data.cpu().numpy()[0]
        if total_gradients is None:
            total_gradients = gradients
        else:
            total_gradients += gradients
    
    smoothed_gradients = total_gradients / num_samples
    smoothed_gradients = np.abs(smoothed_gradients)
    smoothed_gradients = np.mean(smoothed_gradients, axis=0)
    return smoothed_gradients

def correlations(lime_mask, smoothgrad_mask):
    lime_flat = lime_mask.flatten()
    smoothgrad_flat = smoothgrad_mask.flatten()
    lime_norm = (lime_flat - np.min(lime_flat)) / (np.max(lime_flat) - np.min(lime_flat) + 1e-10)
    smoothgrad_norm = (smoothgrad_flat - np.min(smoothgrad_flat)) / (np.max(smoothgrad_flat) - np.min(smoothgrad_flat) + 1e-10)
    kendall_tau, _ = kendalltau(lime_norm, smoothgrad_norm)
    spearman_rho, _ = spearmanr(lime_norm, smoothgrad_norm)
    return kendall_tau, spearman_rho


def plot_explanations(image, lime_mask, smoothgrad_mask, predicted_label, img_name):
    image_resized = image.resize((224, 224))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_resized)
    axes[0].set_title(f'Original\n{predicted_label}')
    axes[0].axis('off')
    lime_mask_resized = resize(lime_mask, (224, 224), order=0, preserve_range=True)
    axes[1].imshow(image_resized)
    axes[1].imshow(lime_mask_resized, cmap='hot', alpha=0.5)
    axes[1].set_title('LIME')
    axes[1].axis('off')
    axes[2].imshow(image_resized)
    axes[2].imshow(smoothgrad_mask, cmap='hot', alpha=0.5)
    axes[2].set_title('SmoothGrad')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(f'explanation_{img_name}', dpi=150, bbox_inches='tight')
    plt.close()

imagenet_path = './imagenet_samples'
image_paths = os.listdir(imagenet_path)

for img_path in image_paths:
    # Open and preprocess the image
    my_img = os.path.join(imagenet_path, img_path)
    input_image = Image.open(my_img).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_synset = idx2synset[predicted_idx]
    predicted_label = idx2label[predicted_idx]

    print(f"\nImage: {img_path}")
    print(f"predicted label: {predicted_synset} ({predicted_label})")

    lime_mask, lime_weights, segments = lime(input_image, model, num_samples=100, num_features=10)
    smoothgrad_mask = smoothgrad_explain(input_image, model, num_samples=50, noise_level=0.15)
    
    lime_mask_resized = resize(lime_mask, smoothgrad_mask.shape, order=0, preserve_range=True)
    
    kendall_tau, spearman_rho = correlations(lime_mask_resized, smoothgrad_mask)
    
    print(f"Kendall's Tau: {kendall_tau:.4f}")
    print(f"Spearman's Rho: {spearman_rho:.4f}")
    plot_explanations(input_image, lime_mask, smoothgrad_mask, predicted_label, img_path.replace('.JPEG', '.png'))
