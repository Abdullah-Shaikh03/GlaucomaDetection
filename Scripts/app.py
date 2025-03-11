import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# Load your trained model
model = torch.load("models/best_model.pth")  # Replace with your model path
model.eval()

# Select the convolutional layer to extract feature maps
target_layer = model.backbone.layer4[-1].conv3  # Adjust based on your architecture

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook the gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]  # Get the target class output
        target.backward()

        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU activation
        heatmap /= np.max(heatmap)  # Normalize

        return heatmap

# Load an example image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

# Generate Grad-CAM
gradcam = GradCAM(model, target_layer)
image_tensor, orig_image = preprocess_image("sample_fundus.jpg")  # Replace with actual image path
heatmap = gradcam.generate_cam(image_tensor, class_idx=1)  # Assuming class 1 is Glaucoma

# Overlay heatmap on original image
heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Blend with original
superimposed_img = cv2.addWeighted(orig_image, 0.6, heatmap, 0.4, 0)

# Show result
plt.imshow(superimposed_img)
plt.axis("off")
plt.title("Grad-CAM Visualization for Optic Cup & Disc")
plt.show()
