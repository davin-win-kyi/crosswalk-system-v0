import torch
from PIL import Image
from torchvision import transforms


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open the image file and ensure it's in RGB format

    transformations = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the size your model expects
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensor
    ])

    return transformations(image).unsqueeze(0)  # Add a batch dimension and return the transformed tensor



model_path = 'C:\\Users\\davin\\PycharmProjects\\Depth_Testing\\best.pt'
model = torch.load(model_path)['model']
model = model.to('cuda').float()
model.eval()  # Set the model to evaluation mode


img_path = 'C:\\Users\\davin\\PycharmProjects\\Depth_Testing\\crosswalk_testimage#1.jpg'
img = preprocess_image(img_path)

with torch.no_grad():  # Disable gradient computation for inference
    prediction = model(img.to('cuda'))
    # Process the prediction as needed

print("PREDICTION: ", prediction)

bbox_tensor = prediction[0][0]  # This might be the tensor containing bounding boxes and class scores

# The exact indices for bounding box coordinates and class scores depend on the model's output format
# Here's an example of how they might be extracted
for i in range(bbox_tensor.size(0)):  # Iterate over each detection
    bbox = bbox_tensor[i, :4]  # Assuming the first 4 values are the bounding box coordinates
    class_score = bbox_tensor[i, 4]  # Assuming the 5th value is the class score/confidence
    class_index = bbox_tensor[i, 5]  # Assuming the 6th value is the class index

    # Do something with the bounding box and class (e.g., print them)
    print(f"Bounding Box: {bbox}, Class Index: {class_index}")