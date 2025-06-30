# Neural Network from Scratch

Simple neural network implementation for MNIST digit classification, built from scratch with NumPy.

*Based on Michael Nielsen's "[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)", adjusted slightly for my [Youtube video](https://youtu.be/WLmY9icEOQk).*

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
2. **Train the network:**

   **Option A: Jupyter Notebook**

   ```bash
   jupyter notebook train.ipynb
   ```

   Run all cells to train and save the model.

   **Option B: Python directly**

   ```python
   import network
   import mnist_loader

   net = network.Network([784, 16, 16, 10]) # adjust as your NN structure
   training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
   net.train(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
   net.save("trained_network.pkl")
   ```
3. **Test with web app:**

   ```bash
   streamlit run app.py
   ```

   Draw digits and see predictions in your browser.

## Files

- `network.py` - Neural network implementation
- `mnist_loader.py` - MNIST data loader
- `train.ipynb` - Training notebook
- `app.py` - Streamlit web interface
- `data/mnist.pkl.gz` - MNIST dataset

## Network Architecture Example

- Input: 784 neurons (28×28 pixels)
- Hidden: 16 → 16 neurons
- Output: 10 neurons (digits 0-9)
- Activation: Sigmoid
