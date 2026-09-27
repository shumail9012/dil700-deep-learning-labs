"""
DIL700 Lab 1 – Tensors and first neural networks (TensorFlow / Keras)

Task 1: basic tensor operations in NumPy
Task 2: a small dense binary classifier (model definition)
Task 3: MLP on MNIST and an embedding model for IMDB sentiment classification
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# TASK 1 
tensor1 = np.array([[1, 2, 3], [4, 5, 6]])
tensor2 = np.array([[10, 20, 30], [40, 50, 60]])
print("1. Tensor 1:\n", tensor1)
print("   Tensor 2:\n", tensor2)

addition = tensor1 + tensor2
print("\n2. Addition:\n", addition)

multiplication = tensor1 * tensor2
print("\n3. Multiplication:\n", multiplication)

reshaped = np.arange(12).reshape(4, 3)
print("\n4. Reshape (4x3):\n", reshaped)

sliced_value = tensor1[0, 1] 
print("\n5. Slicing (Row 0, Col 1):", sliced_value)

combined = np.concatenate((tensor1, tensor2), axis=0)
print("\n6. Combining:\n", combined)

split_tensors = np.array_split(combined, 2)
print("\n7. Splitting (First part):\n", split_tensors[0])

mean_value = np.mean(tensor1)
print("\n8. Mean value:", mean_value)

#TASK 2 

model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(8,)),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


#TASK 3

# for MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#Normalize the images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(10, activation='softmax') 
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

# Some visualization of training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print("-" * 30)
print(f"8. Final Test Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("-" * 30)

#for IMDB dataset
vocab_size = 10000
max_length = 256

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
train_data = pad_sequences(train_data, maxlen=max_length)
test_data = pad_sequences(test_data, maxlen=max_length)
model = models.Sequential([
    layers.Embedding(vocab_size, 16, input_length=max_length),
    
    layers.GlobalAveragePooling1D(),

    layers.Dense(16, activation='relu'),
    
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary crossentropy for 2 classes
              metrics=['accuracy'])
history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_split=0.2, # Use 20% of data for validation
                    verbose=1)
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

# Some visualization of training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, 'bo', label='Training Loss') 
plt.plot(epochs_range, val_loss, 'b', label='Validation Loss') 
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, 'ro', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
print("-" * 30)
print(f"8. Final Test Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("-" * 30)

