import io
import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from torchvision.models import resnet50, ResNet50_Weights
import pinecone

app = Flask(__name__)

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index_name = "fashion-ai-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=2048, metric="cosine")
vector_db = pinecone.Index(index_name)

# Initialize the model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
model.fc = torch.nn.Linear(model.fc.in_features, 1000)  # Adjust the output size as needed
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    with torch.no_grad():
        features = feature_extractor(image)
    return features.squeeze().numpy()

@app.route('/train', methods=['POST'])
def train():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    images = request.files.getlist('images')
    labels = request.form.getlist('labels')

    if len(images) != len(labels):
        return jsonify({'error': 'Number of images and labels must match'}), 400

    # Process and add new images to the vector database
    vectors_to_upsert = []
    tensors_to_train = []
    labels_to_train = []

    for i, (image, label) in enumerate(zip(images, labels)):
        img = Image.open(io.BytesIO(image.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Extract features and store in vector database
        features = extract_features(img_tensor)
        vector_id = f"img_{len(vectors_to_upsert)}"
        vectors_to_upsert.append((vector_id, features.tolist(), {"label": int(label)}))
        
        # Prepare tensors for training
        tensors_to_train.append(img_tensor)
        labels_to_train.append(int(label))

    # Upsert vectors to Pinecone
    vector_db.upsert(vectors=vectors_to_upsert)

    # Train the model
    model.train()
    tensors = torch.cat(tensors_to_train)
    labels = torch.tensor(labels_to_train)

    for epoch in range(5):  # Adjust the number of epochs as needed
        optimizer.zero_grad()
        outputs = model(tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return jsonify({'message': 'Training completed successfully'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    img = Image.open(io.BytesIO(image.read())).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    # Extract features
    features = extract_features(img_tensor)

    # Query vector database for similar images
    query_result = vector_db.query(features.tolist(), top_k=5, include_metadata=True)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probabilities, 5)

    model_predictions = [
        {'class': int(class_idx), 'probability': float(prob)}
        for prob, class_idx in zip(top_prob[0], top_class[0])
    ]

    # Combine vector database results and model predictions
    similar_images = [
        {'id': match['id'], 'score': match['score'], 'label': match['metadata']['label']}
        for match in query_result['matches']
    ]

    return jsonify({
        'model_predictions': model_predictions,
        'similar_images': similar_images
    }), 200

if __name__ == '__main__':
    app.run(debug=True)