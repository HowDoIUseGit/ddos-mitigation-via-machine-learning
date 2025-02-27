import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import preprocess_data
import pandas as pd

print("================== Starting Model Training ==================")

print("Loading dataset...")
data = pd.read_csv('/home/shajek/antiDDoSiwthML/ddosMitigationSystemWithML/final_dataset.csv')

print("Preprocessing data...")
X_scaled, y, label_mapping, scaler, valid_indices = preprocess_data(data)

print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("Building the model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='leaky_relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("Compiling the model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training the model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    verbose=1
)

print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

print("Saving evaluation metrics to 'evaluation_metrics_v3.txt'")
with open('evaluation_metrics_v3.txt', 'w') as eval_file:
    eval_file.write(f"Test Loss: {test_loss}\n")
    eval_file.write(f"Test Accuracy: {test_accuracy}\n")
print("Evaluation metrics saved to 'evaluation_metrics_v3.txt'")

print("Generating classification report...")
y_pred = (model.predict(X_test) > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
class_report = classification_report(y_test, y_pred, target_names=label_mapping.values())
print("Classification Report:\n", class_report)

print("Saving classification report to 'classification_report_v3.txt'")
with open('classification_report_v3.txt', 'w') as report_file:
    report_file.write("Confusion Matrix:\n")
    report_file.write(str(conf_matrix))
    report_file.write("\n\nClassification Report:\n")
    report_file.write(class_report)
print("Classification report saved to 'classification_report_v3.txt'")

print("Saving the model...")
model.save('ddos_recognition_model_v3.h5')
print("Model saved as 'ddos_recognition_model_v3.h5'")

print("================== Finished Model Training ==================")
