# MNIST Digit Classifier
Sieć konwolucyjna (CNN) do rozpoznawania cyfr z datasetu MNIST.
## Architektura modelu
Convolutional layer 1
- Warstwa Conv2D 
- Funkcja ReLU
- MaxPooling 2x2

Convolutional layer 2
- Warstwa Conv2D 
- Funkcja ReLU
- MaxPooling 2x2

Klasyfikator:
- Flatten
- Warstwa Linear
- ReLU
- Dropout
- Warstwa Linear
  
## Wyniki
Epoch: 1 | train loss: 0.2146 | test loss: 0.0619 | accuracy: 97.90%
Epoch: 2 | train loss: 0.0677 | test loss: 0.0340 | accuracy: 98.83%
Epoch: 3 | train loss: 0.0473 | test loss: 0.0323 | accuracy: 98.94%
Epoch: 4 | train loss: 0.0382 | test loss: 0.0300 | accuracy: 98.90%
Epoch: 5 | train loss: 0.0325 | test loss: 0.0353 | accuracy: 98.86%
Epoch: 6 | train loss: 0.0259 | test loss: 0.0265 | accuracy: 99.22%
Epoch: 7 | train loss: 0.0219 | test loss: 0.0312 | accuracy: 99.03%
Epoch: 8 | train loss: 0.0198 | test loss: 0.0266 | accuracy: 99.19%
Epoch: 9 | train loss: 0.0171 | test loss: 0.0244 | accuracy: 99.22%
