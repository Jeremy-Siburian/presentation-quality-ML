import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_curve, auc, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Path to the directory containing the extracted ROIs
roi_dir = "/Users/jeremydasa/Desktop/MEF_E_Master/Extracted_Faces"

# Output directory for the ROC curve figure
output_dir = "/Users/jeremydasa/Desktop/MEF_E_Master/ROC_Plots"

# Load ROIs and labels
rois = []
labels = []

for label in ["Attentive", "Distractive"]:
    label_dir = os.path.join(roi_dir, label)
    for roi_file in os.listdir(label_dir):
        roi_path = os.path.join(label_dir, roi_file)
        roi_image = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        roi_image = cv2.resize(roi_image, (64, 64))  # Resize the frame to match the model input size
        rois.append(roi_image)
        labels.append(label)

# Convert lists to numpy arrays
rois = np.array(rois)
labels = np.array(labels)

# Encode labels as binary values (0 for "Attentive" and 1 for "Distractive")
labels_encoded = (labels == "Distractive").astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(rois, labels_encoded, test_size=0.2, random_state=42)

# Normalize the pixel values to a range of [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data to match the 3D CNN input shape (num_samples, frames=1, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 1, 64, 64, 1)

# Convert labels to categorical format (one-hot encoding)
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Create the 3D CNN model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(1, 64, 64, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions on the test set
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Compute evaluation metrics
precision = precision_score(np.argmax(y_test, axis=1), y_pred_labels)
recall = recall_score(np.argmax(y_test, axis=1), y_pred_labels)
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_labels)

print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1-Score:", f1)

# Generate and save the ROC curve
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred_probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Save the ROC curve figure
roc_curve_path = os.path.join(output_dir, "roc_curve_3DConv.png")
plt.savefig(roc_curve_path)

# Save the trained model for future predictions
model.save(os.path.join(output_dir, "lecture_quality_model_3d_cnn_single_frame.h5"))
