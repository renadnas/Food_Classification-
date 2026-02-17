# Food_Classification-
computer vision pipeline that classifies food images into 8 distinct categories and provides estimated calorie counts. This project leverages Transfer Learning by using a pre-trained ResNet50 model to extract high-level features, which are then classified using Support Vector Machines (SVM), Multi-Layer Perceptrons (MLP), and Random Forest.
Overview
The system follows a three-stage process:
Feature Extraction: Images are processed through a ResNet50 model (pre-trained on ImageNet) with the top layer removed, converting pixels into a 2048-dimensional feature vector.
Classification: These vectors are fed into classical ML algorithms to identify the food type.
Deployment: A dedicated script allows users to input a single image to receive the top 3 predictions and a calorie estimation based on the primary result.

Dataset
The project uses the Food-11 Image Dataset, filtered for the following classes:
Bread, Dairy product, Dessert, Egg, Fried Food, Meat, Rice, Seafood.
Sample Size: 300 images per class for training, validation, and testing to ensure balanced evaluation.
Tech Stack
Deep Learning: TensorFlow/Keras (ResNet50)
Machine Learning: Scikit-learn (SVM, MLP, Random Forest)
Data Handling: NumPy, Pandas, Kagglehub
Visualization: Matplotlib, Seaborn
Project Structure
food_classification.py: The core training script. It downloads the data, filters it, extracts features, trains three different models, and evaluates them with confusion matrices.
food_model_deployment.py: The inference script. Loads the saved SVM model to predict classes for new images and estimates calories.
svm_model.pkl: The serialized trained SVM model.
class_indices.json: Mapping of class names to numeric labels.

Model Performance
The training script generates comprehensive metrics for all three classifiers, including:
Accuracy, Precision, Recall, and F1-Score.
AUC-ROC Curves for multi-class evaluation.
Confusion Matrices to identify which food groups are most frequently confused (e.g., Meat vs. Fried Food).
How to Run
1. Requirements
Bash
pip install tensorflow scikit-learn numpy tqdm kagglehub matplotlib seaborn

2. Training
Run the classification script to download the dataset and train the models:
Bash
python food_classification.py

3. Inference & Calorie Prediction
Update the IMAGE_PATH in food_model_deployment.py to point to your image, then run:
Bash
python food_model_deployment.py

Calorie Estimation Logic
The system provides a calorie estimate per 100g serving based on the top prediction: | Food Category | Calories (approx. per 100g) | Bread | 265 kcal | | Meat | 250 kcal | | Dessert | 350 kcal | | Rice | 130 kcal |

