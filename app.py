from flask import Flask, request, jsonify
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from keras.applications import ResNet50V2, InceptionResNetV2
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input

app = Flask(__name__)

# Load model sekali di awal
model_incep = InceptionResNetV2(input_shape=(299, 299, 3), weights='imagenet', include_top=False)
model_resnet = ResNet50V2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

def preprocess_image(img_bytes, target_size):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, target_size)
    return img

def extract_lbp_features(img):
    lbp = cv2.calcHist([img], [0], None, [256], [0, 256])
    lbp = lbp.flatten()
    lbp = lbp / (lbp.sum() + 1e-8)
    return lbp

def extract_cnn_features(img, model, preprocess_func):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    features = model.predict(img_array)
    return features.flatten()

def binarize_features(features, threshold=0.05):
    return (features > threshold).astype(int)

@app.route("/", methods=["GET"])
def hello():
    return("hello")

@app.route("/calc", methods=["POST"])
def calculate_grade():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Both image1 and image2 are required"}), 400

    img1_bytes = request.files["image1"].read()
    img2_bytes = request.files["image2"].read()

    # Preprocessing
    img1_incep = preprocess_image(img1_bytes, (299, 299))
    img2_incep = preprocess_image(img2_bytes, (299, 299))
    img1_resnet = preprocess_image(img1_bytes, (224, 224))
    img2_resnet = preprocess_image(img2_bytes, (224, 224))

    # Feature extraction
    lbp1 = extract_lbp_features(img1_incep)
    lbp2 = extract_lbp_features(img2_incep)
    cnn1 = extract_cnn_features(img1_incep, model_incep, preprocess_input)
    cnn2 = extract_cnn_features(img2_incep, model_incep, preprocess_input)

    lbp3 = extract_lbp_features(img1_resnet)
    lbp4 = extract_lbp_features(img2_resnet)
    cnn3 = extract_cnn_features(img1_resnet, model_resnet, preprocess_input)
    cnn4 = extract_cnn_features(img2_resnet, model_resnet, preprocess_input)

    combined1 = np.hstack((lbp1, cnn1))
    combined2 = np.hstack((lbp2, cnn2))
    combined3 = np.hstack((lbp3, cnn3))
    combined4 = np.hstack((lbp4, cnn4))

    bin1 = binarize_features(combined1)
    bin2 = binarize_features(combined2)
    bin3 = binarize_features(combined3)
    bin4 = binarize_features(combined4)

    # Similarity calculations
    sim_cos_incep = float(cosine_similarity([combined1], [combined2])[0][0])
    sim_jac_incep = float(jaccard_score(bin1, bin2, average="binary"))
    sim_cos_resnet = float(cosine_similarity([combined3], [combined4])[0][0])
    sim_jac_resnet = float(jaccard_score(bin3, bin4, average="binary"))

    return jsonify({
        "similarity_incep_cosine": sim_cos_incep,
        "similarity_incep_jaccard": sim_jac_incep,
        "similarity_resnet_cosine": sim_cos_resnet,
        "similarity_resnet_jaccard": sim_jac_resnet
    })

if __name__ == "__main__":
    app.run(port =8080)
