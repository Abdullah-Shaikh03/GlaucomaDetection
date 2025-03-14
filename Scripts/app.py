import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# Import your model class
from Model import get_model  # Make sure this imports correctly

# Initialize the model with the correct architecture
model = get_model("cuda")

# Load the saved weights - explicitly set weights_only to False
try:
    checkpoint = torch.load("models/model_state.pth", weights_only=False)
    # Extract the model state dictionary from the checkpoint
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully using model_state_dict key")
    else:
        print("Unexpected checkpoint format. Keys found:", checkpoint.keys())
        raise ValueError("Could not find model_state_dict in checkpoint")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()

model.eval()

# Select the appropriate layer for visualization
# Based on your HybridNet architecture:
target_layer = model.layer4[-1].se  # Using the SEBlock in the last HybridBlock of layer4

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if len(output.shape) == 1:
            output = output.unsqueeze(0)  # Add batch dimension if needed
            
        target = output[0, class_idx]  # Get the target class output
        target.backward()
        
        # Compute Grad-CAM
        if self.gradients is None:
            print("Warning: No gradients detected. Check if backpropagation reached the target layer.")
            return np.zeros((1, 1))  # Return empty heatmap
            
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU activation
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)  # Normalize with epsilon
        
        return heatmap

# Define preprocessing function
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Keep a copy of original image for visualization
        orig_image = image.copy()
        
        # Apply proper preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor.to("cuda"), orig_image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

# Generate Grad-CAM
try:
    gradcam = GradCAM(model, target_layer)
    
    # Replace with your actual image path
    image_path = "../image.png"
    print(f"Loading image from: {image_path}")
    
    image_tensor, orig_image = preprocess_image(image_path)
    print("Image loaded and preprocessed successfully")
    
    print("Generating Grad-CAM...")
    heatmap = gradcam.generate_cam(image_tensor, class_idx=1)  # Assuming class 1 is Glaucoma
    print("Grad-CAM generated successfully")

    # Overlay heatmap on original image
    heatmap_resized = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_jet = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    # Convert from BGR to RGB
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    # Blend with original image
    superimposed_img = cv2.addWeighted(orig_image, 0.6, heatmap_jet, 0.4, 0)

    # Show result
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(orig_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title("Activation Map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM: Glaucoma Detection")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("analysis_results/gradcam_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualization saved as 'gradcam_visualization.png'")
    
except Exception as e:
    print(f"Error during Grad-CAM generation: {e}")
    import traceback
    traceback.print_exc()