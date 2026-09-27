# Deep Learning Labs (DIL700)

Six hands-on labs from the **DIL700 Deep Learning** course in the M.Sc. programme in AI and Automation at University West, Sweden (Spring 2026). They go from tensors and first networks to training deep networks, CNNs, transfer learning and recurrent networks, all built with **TensorFlow / Keras**.

| Lab | Topics | Highlights |
|-----|--------|------------|
| [1 – Tensors and first networks](lab1-tensors-and-first-networks/) | Tensor ops, dense networks, embeddings | MNIST MLP ~97 % val. accuracy · IMDB sentiment ~87 % |
| [2 – PCA and MLPs](lab2-pca-and-mlps/) | PCA from scratch, kernel PCA, classification and regression MLPs | Manual PCA matches scikit-learn · Kernel PCA + MLP grid search 96.5 % |
| [3 – Training deep networks](lab3-training-deep-networks/) | LR finder, activations and initializers, batch norm, optimizers | MNIST 97.8 % test · SELU trains a 100-layer net where sigmoid fails |
| [4 – Regularization and custom models](lab4-regularization-and-custom-models/) | L1/L2, max-norm, dropout, augmentation, multi-output, bagging, adversarial examples | Two-headed CIFAR-10 model · learned adversarial noise fools the classifier |
| [5 – CNNs and transfer learning](lab5-cnns-and-transfer-learning/) | Convolution and pooling, CNNs, ResNet-34, Xception | Fashion-MNIST CNN 91.8 % · Xception transfer learning 88.4 % on tf_flowers |
| [6 – RNNs and sequences](lab6-rnns-and-sequences/) | SimpleRNN, LSTM, GRU, normalization, sequence generation | 10-step time-series forecasting · Bach chorale generation with LSTMs |

The course's term project, SAR image super-resolution with UNet and ESRGAN, has its own repository: [sar-super-resolution](https://github.com/shumail9012/sar-super-resolution).

## Running the code

```bash
pip install -r requirements.txt
python lab1-tensors-and-first-networks/lab1.py
jupyter notebook lab4-regularization-and-custom-models/lab4.ipynb
```

All datasets (MNIST, Fashion-MNIST, CIFAR-10, IMDB, tf_flowers, JSB Chorales) download automatically through Keras or TensorFlow Datasets. The notebooks are saved with their outputs, so you can read the results without running anything. Labs 4–6 take a few minutes each on a CPU. The Xception part of Lab 5 is much faster on a GPU.

## Team

Group 9 lab work by **Shumail Alam Khan**, Halit Yigit Atalay and Atakan Güdüllüoğlu.
