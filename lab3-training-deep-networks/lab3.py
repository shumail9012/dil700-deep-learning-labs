"""
DIL700 Lab 3 – Training deep neural networks

Task 1: learning-rate finder + MNIST MLP with TensorBoard and early stopping
Task 2: 100-layer networks on Fashion-MNIST with different activations / initializers
Task 3: 20-layer CIFAR-10 networks: batch normalization and optimizer comparison
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir(name):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, name + "_" + run_id)

class LRFinder(keras.callbacks.Callback):
    def __init__(self, min_lr=1e-5, max_lr=10, steps=100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = steps
        self.iteration = 0
        self.history = {}
        
    def on_train_begin(self, logs=None):
        self.iteration = 0
        self.model.optimizer.learning_rate.assign(self.min_lr)
        
    def on_train_batch_end(self, batch, logs=None):
        self.iteration += 1
        factor = (self.max_lr / self.min_lr) ** (1 / self.total_steps)
        lr = self.min_lr * (factor ** self.iteration)
        self.model.optimizer.learning_rate.assign(lr)
        
        loss = logs["loss"]
        if "lr" not in self.history: self.history["lr"] = []
        if "loss" not in self.history: self.history["loss"] = []
        self.history["lr"].append(lr)
        self.history["loss"].append(loss)

    def plot_loss(self):
        plt.plot(self.history["lr"], self.history["loss"])
        plt.xscale('log')
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True)
        plt.show()

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

def build_mnist_model():
    model = keras.models.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    return model

model_lr = build_mnist_model()
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
model_lr.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

lr_finder = LRFinder(min_lr=1e-3, max_lr=1, steps=200) 
model_lr.fit(X_train, y_train, epochs=1, steps_per_epoch=200, callbacks=[lr_finder], verbose=0)

lr_finder.plot_loss()

optimal_lr = 0.2  
print(f"Training final model with LR: {optimal_lr}")

model = build_mnist_model()
optimizer = keras.optimizers.SGD(learning_rate=optimal_lr)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir("mnist_task1"))
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb, early_stopping_cb])
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"MNIST Test Accuracy: {test_acc:.4f}")

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

activations = ["sigmoid", "relu", "elu", "selu"]

for act in activations:
    print(f"\nTesting Activation: {act}")
    
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())
    
    for _ in range(100):
        if act == "selu":
            model.add(keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
        elif act == "relu" or act == "elu":
            model.add(keras.layers.Dense(100, activation=act, kernel_initializer="he_normal"))
        else:
            model.add(keras.layers.Dense(100, activation=act)) 
            
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=1e-3), metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, epochs=3, validation_data=(X_valid, y_valid), verbose=1)
    
    print(f"Result for {act}: Last Val Loss: {history.history['val_loss'][-1]:.4f}")

print("\n--- STARTING TASK 3: CIFAR10 ---")

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train_full = X_train_full / 255.0 
X_test = X_test / 255.0
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

def build_cifar_model(use_bn=False):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(32, 32, 3)))
    model.add(keras.layers.Flatten())
    
    for _ in range(20):
        if use_bn:
            model.add(keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("elu"))
        else:
            model.add(keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"))
            
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model

print("\nComparing Model with and without Batch Normalization...")
configs = [("Without BN", False), ("With BN", True)]

for name, use_bn in configs:
    print(f"Training {name}...")
    model = build_cifar_model(use_bn=use_bn)
    
    optimizer = keras.optimizers.Nadam(learning_rate=5e-4) 
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir(f"cifar_{name.replace(' ', '_')}"))
    
    model.fit(X_train, y_train, epochs=20, 
              validation_data=(X_valid, y_valid),
              callbacks=[early_stopping, tensorboard_cb], verbose=1)

optimizers_dict = {
    "Momentum": keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
    "Nesterov": keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True),
    "AdaGrad": keras.optimizers.Adagrad(learning_rate=0.01),
    "RMSProp": keras.optimizers.RMSprop(learning_rate=1e-3),
    "Adam": keras.optimizers.Adam(learning_rate=1e-3),
    "Nadam": keras.optimizers.Nadam(learning_rate=1e-3)
}

for opt_name, opt_obj in optimizers_dict.items():
    print(f"Testing Optimizer: {opt_name}")
    model = build_cifar_model(use_bn=False)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt_obj, metrics=["accuracy"])
    
    early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir(f"cifar_opt_{opt_name}"))
    
    history = model.fit(X_train, y_train, epochs=15, 
                        validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping, tensorboard_cb], verbose=0)
    
    print(f"{opt_name} - Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")

print("\nAll tasks completed.")