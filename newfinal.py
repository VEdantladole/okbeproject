import os
import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import secrets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications import imagenet_utils
from utils import resize_img
import torch
import torch.nn as nn
from torchvision import transforms


# Flask App Configuration and Database Setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///my.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = secrets.token_hex(16)
db = SQLAlchemy(app)

# Define the User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    lastname = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(15), nullable=False)

# Image Preprocessing
test_dir = 'combined_dataset_toupload'
# test_dir = 'DATSETFINALTOUPLOAD'
currency_dir = 'currency_images'
MIN_MATCH_COUNT = 5
KERNEL_SIZE = 11

# Function to preprocess images
def preprocess(img, showImages=True):
    if showImages:
        cv2.imshow('Before Processing', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Resize the image by reducing the size by 30%
    img = resize_img(img, 0.7) 

    if showImages:
        cv2.imshow('After Resize', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if showImages:
        cv2.imshow('After Grayscale', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img = cv2.bilateralFilter(img, KERNEL_SIZE, KERNEL_SIZE * 2, KERNEL_SIZE // 2)

    if showImages:
        cv2.imshow('After Bilateral Blur', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if showImages:
        cv2.imshow('After Otsu thresholding', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

    if showImages:
        cv2.imshow('After Morphological Processing', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


#MODEL PREDICTIONS START HERE
IMG_WIDTH, IMG_HEIGHT = 224, 224
CONFIDENCE_THRESHOLD = 0.6  # Adjust this threshold as needed

# Load the trained models
alexnet_model = load_model("alexnet_model.h5")
resnet_model = load_model("resnet18_model.h5")
rnn_model = load_model("your_model.h5")


# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)

# Function to classify an image as fake or real using AlexNet
def classify_image_alexnet(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = alexnet_model.predict(preprocessed_image)
    confidence = prediction[0][0] if prediction[0][0] > prediction[0][1] else prediction[0][1]
    if confidence > CONFIDENCE_THRESHOLD:
        label = "Real" if prediction[0][0] > prediction[0][1] else "Fake"
        return label, confidence
    else:
        return "Uncertain", confidence

# Function to classify an image as fake or real using ResNet
def classify_image_resnet(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = resnet_model.predict(preprocessed_image)
    confidence = prediction[0][0] if prediction[0][0] > prediction[0][1] else prediction[0][1]
    print(prediction[0][0])
    print(prediction[0][1])
    
    if confidence > CONFIDENCE_THRESHOLD:
        label = "Real" if prediction[0][0] > prediction[0][1] else "Fake"
        return label, confidence
    else:
        return "Uncertain", confidence


def preprocess_image_for_rnn(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, (200, 200))  # Resize image to match the expected input shape of the RNN model
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to classify an image as fake or real using RNN
def classify_image_rnn(image_path):
    preprocessed_image = preprocess_image_for_rnn(image_path)
    prediction = rnn_model.predict(preprocessed_image)
    confidence = np.max(prediction[0])  # Get the highest confidence score
    label = "Real" if prediction[0][0] > 0.5 else "Fake"  # Assuming binary classification with a sigmoid output
    # label = "Real" if prediction[0][0] > prediction[0][1] else "Fake"
    return label, confidence
    
    

# Flask Routes
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    # Implementation for user registration
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']
    name = request.form['name']
    lastname = request.form['lastname']
    phone = request.form['phone']

    # Check if the username or email is already registered
    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return render_template('wrong.html', message='Username or email already exists.') #change kelay

    # Create a new user and add it to the database
    new_user = User(username=username, password=password, email=email, name=name, lastname=lastname, phone=phone)
    db.session.add(new_user)
    db.session.commit()  #commits the changes to the database

    return redirect(url_for('index')) #index.html is called

@app.route('/login', methods=['POST'])
def login():
    # Implementation for user login
    username = request.form['username']
    password = request.form['password']

    # Check if the user exists and the password is correct
    user = User.query.filter_by(username=username, password=password).first()

    if user:
        # Store user information in the session
        session['user_id'] = user.id
        flash('Login successful!', 'success') 
        return redirect(url_for('index'))
    else:
        return render_template('loginwrong.html', message='Invalid username or password.')

@app.route('/logout')
def logout():
    # Implementation for user logout
    # Remove the user information from the session
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    # Implementation for image upload
    if 'user_id' not in session:
        flash('You need to log in to upload an image.', 'error')
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        test_image_name = file.filename
        file.save(os.path.join(test_dir, test_image_name))
        return redirect(url_for('result', test_image_name=test_image_name))


@app.route('/result/<test_image_name>')
def result(test_image_name):
    if 'user_id' not in session:
        flash('You need to log in to view the result.', 'error')
        return redirect(url_for('login'))
    
    print('Currency Recognition Program starting...\n')
    print('Actual Denomination', Path(test_image_name).stem)

    training_set = [
        img for img in glob.glob(os.path.join(currency_dir, "*.jpg"))
    ]
    training_set_name = [
        Path(img_path).stem for img_path in training_set
    ]

    test_image_loc = os.path.join(test_dir, test_image_name)
    test_img = cv2.imread(test_image_loc)

    # preprocess image
    test_img = preprocess(test_img)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(test_img, mask=None)

    max_matches = -1
    sum_good_matches = 0
    kp_perc = 0

    for i in range(len(training_set)):
        train_img = cv2.imread(training_set[i])
        train_img = preprocess(train_img, showImages=False)
        kp2, des2 = orb.detectAndCompute(train_img, mask = None)

        # brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Match descriptors
        all_matches = bf.knnMatch(des1, des2, k=2)

        good = []

        # store all the good matches as per Lowe's ratio test.
        for m, n in all_matches:
            if m.distance < 0.6 * n.distance:
                good.append([m])

        num_matches = len(good)
        sum_good_matches += num_matches

        if num_matches > max_matches:
            max_matches = num_matches
            best_i = i
            best_kp = kp2
            max_good_matches = len(good)
            best_img = train_img

        print(f'{i+1} \t {training_set[i]} \t {len(good)}')

    kp_perc = (max_good_matches/sum_good_matches*100) if sum_good_matches > 0 else 0

    if max_matches >= MIN_MATCH_COUNT and (kp_perc >= 40):
        print(f'\nMatch Found!\n{training_set_name[best_i]} has maximum matches of {max_matches} ({kp_perc}%)')

        match_img = cv2.drawMatchesKnn(test_img, kp1, best_img, best_kp, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        note = training_set_name[best_i]
        print(f'\nDetected denomination: {note}')

        plt.imshow(match_img), plt.title(f'DETECTED MATCH!! Hence the Currency is true: {note}'), plt.show()

    else:
        print(f'\nNo Good Matches, closest one has {max_matches} matches ({kp_perc}%)')

        closest_match = cv2.drawMatchesKnn(test_img, kp1, best_img, best_kp, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        note = training_set_name[best_i]

        plt.imshow(closest_match), plt.title('NO MATCH!! Hence Currency is Fake'), plt.show()

    print('\nProgram exited')

    # Classification using AlexNet model
    alexnet_classification, alexnet_confidence = classify_image_alexnet(test_image_loc)

    # Classification using ResNet model
    resnet_classification, resnet_confidence = classify_image_resnet(test_image_loc)

    rnn_classification, rnn_confidence = classify_image_rnn(test_image_loc)

    return render_template('newresult.html', 
        classification_alexnet=alexnet_classification, 
        confidence_alexnet=alexnet_confidence,
        classification_resnet=resnet_classification,
        confidence_resnet=resnet_confidence,
        classification_rnn=rnn_classification,
        confidence_rnn=rnn_confidence)



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

