from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import json

from extensions import db
from models import User, Prediction
from database import init_db

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///kidney_classifier.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db.init_app(app)

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Make datetime available in templates
@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model configuration
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
MODEL_PATH = 'resnet18_kidney_best.pth'

# Image preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Global model variable
model = None
device = None

def load_model():
    """Load the trained ResNet model"""
    global model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    try:
        # Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

class SimpleGradCAM:
    """Simple GradCAM implementation"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        # Reset
        self.gradients = None
        self.activations = None
        
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), class_idx
    
    def cleanup(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def create_gradcam_overlay(image, heatmap, alpha=0.6):
    """Create GradCAM overlay"""
    # Denormalize image
    img_array = image.permute(1, 2, 0).cpu().numpy()
    img_array = img_array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_array = np.clip(img_array, 0, 1)
    
    # Resize heatmap to image size
    H, W = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # Apply threshold (top 15% activations)
    threshold = np.percentile(heatmap_resized, 85)
    mask = heatmap_resized >= threshold
    
    # Create colored heatmap
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    
    # Create overlay
    overlay = img_array.copy()
    mask_3d = np.stack([mask, mask, mask], axis=2)
    overlay = (1 - alpha * mask_3d) * overlay + alpha * mask_3d * heatmap_colored
    
    return np.clip(overlay, 0, 1)

def predict_image(image_path):
    """Predict kidney condition and generate GradCAM"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probabilities = probabilities.squeeze().cpu().numpy()
        class_probabilities = {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(all_probabilities)}
        
        # Generate GradCAM
        gradcam = SimpleGradCAM(model, model.layer3[-1].conv2)  # Using layer 3 as optimal
        heatmap, _ = gradcam.generate(input_tensor, predicted_idx.item())
        
        # Create overlay
        overlay = create_gradcam_overlay(input_tensor.squeeze(), heatmap)
        
        # Convert overlay to base64 for web display
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        original_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        original_img = original_img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        original_img = np.clip(original_img, 0, 1)
        plt.imshow(original_img)
        plt.title('Original CT Scan')
        plt.axis('off')
        
        # GradCAM overlay
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f'GradCAM: {predicted_class}\nConfidence: {confidence_score:.2f}')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Convert to base64
        overlay_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Cleanup
        gradcam.cleanup()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'class_probabilities': class_probabilities,
            'overlay_image': overlay_base64,
            'success': True
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('signup.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).limit(10).all()
    
    return render_template('dashboard.html', predictions=predictions)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_image(filepath)
            
            if result['success']:
                # Save prediction to database
                prediction = Prediction(
                    user_id=session['user_id'],
                    filename=filename,
                    predicted_class=result['predicted_class'],
                    confidence=result['confidence'],
                    class_probabilities=json.dumps(result['class_probabilities'])
                )
                
                db.session.add(prediction)
                db.session.commit()
                
                return render_template('result.html', 
                                     result=result, 
                                     filename=filename,
                                     prediction_id=prediction.id)
            else:
                flash(f'Prediction failed: {result["error"]}')
                os.remove(filepath)  # Clean up failed upload
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, TIFF, or DCM files.')
    
    return render_template('predict.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
    
    return render_template('history.html', predictions=predictions)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'})
    
    # Save temporary file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"api_{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Make prediction
    result = predict_image(filepath)
    
    # Clean up temporary file
    os.remove(filepath)
    
    return jsonify(result)

if __name__ == '__main__':
    # Initialize database and load model on startup
    with app.app_context():
        init_db(app)
        print("✅ Database initialized")
    
    # Load model
    if not load_model():
        print("⚠️ Warning: Model not loaded. Predictions will not work.")
        print("Make sure 'resnet18_kidney_best.pth' exists in the project directory")
    
    # Run app
    print("🚀 Starting Flask server...")
    print("📊 Visit http://localhost:5000 to access the application")
    app.run(debug=True,  port=5000)