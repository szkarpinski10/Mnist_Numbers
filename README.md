# MNIST Digit Classifier
Sieć konwolucyjna (CNN) do rozpoznawania cyfr z datasetu MNIST.
## Architektura modelu

**Convolutional layer 1**
- Warstwa Conv2D 
- Funkcja ReLU
- MaxPooling 2x2

**Convolutional layer 2**
- Warstwa Conv2D 
- Funkcja ReLU
- MaxPooling 2x2

**Klasyfikator:**
- Flatten
- Warstwa Linear
- ReLU
- Dropout
- Warstwa Linear
