# Machine-Learning-Java
A feed-through network structure designed in Java.

This project consists of three files:
- NeuralNetwork.java
- Neuron.java
- XORExample.java

These work in tandem, and should be able to provide for most tasks any FFN would require - including backpropagation and mini-batching.

## Setting Up a Neural Network
In order to set up a Neural Network, create an instance of the NeuralNetwork class, providing the number of inputs for the NN, the number of outputs, the number of hidden layers, an int[] array specifying the size of each hidden layer, a double for the learning rate, and a double for the momentum.
```java
NeuralNetwork nn = new NeuralNetwork(2, 1, 1, new int[] {4}, 0.5, 0.9);
```
Then, initialize the NN.
```java
nn.initialize();
```
Of note: If you wish to set the specific activation type for a layer of neurons, call setLayerActivation() after the NeuralNetwork is initialized.
```java
nn.setLayerActivation(int layer, int activationFunction);
```
Otherwise, the NN's hidden neurons use the ReLU activation function and outputs use an efficient Sigmoid approximate (x / 1 + abs(x)).

## Setting Inputs
Use the setInputs(double[] inputs) method to set the value of the input layer's neurons.
```java
nn.setInputs(new double[] {random.nextInt(2), random.nextInt(2)};
```

## Forward and Back Propagation
Once a NeuralNetwork has been initialized and had inputs set, use the calculate() function to perform a forwards propagation through the network - this is not necessary if you intend to call the backProp() method.
```java
nn.calculate();
nn.backProp(expectedOutput);
```
You may also get the output of the NeuralNetwork with the getOutputs() method.
```java 
double[] outputs = nn.getOutputs;
```

## Mini-batch Capability
By using the sumError() method, you can iterate over multiple examples and sum the error of those examples, which can then be used by the backPropSumError() method to back propagate the average of that error.
```java
for (int i = 0; i < outputs.length; i++) {
    nn.sumError(outputs[i], expectedOutputs[i]);
}
nn.backPropSumError();
```
This process keeps track of the number of sumError calls, and returns the value to 0 at the end of the process.
If you're interested, try implementing mini-batching into the XOR Example below.

## XOR Example
Putting this all together, we can reach an XOR approximate rather quickly (2000-3000 epochs).
```java
// Create and initialize the NeuralNetwork.
NeuralNetwork nn = new NeuralNetwork(2, 1, 1, new int[] {4}, 0.2, 0.8);
nn.initialize();

// Create variables to test the network.
double loss = 1.0;
double sessionLength = 50;
double[] inputs;
double[] outputs;
double[] expectedOutputs = new double[1];
int epochs = 0;
Random random = new Random();

// Create and initialize the NeuralNetwork.
NeuralNetwork nn = new NeuralNetwork(2, 1, 1, new int[] {4}, 0.1, 0.9);
nn.initialize();

// Create variables to test the network.
double loss = 1.0;
double sessionLength = 50;
double[] inputs;
double[] outputs;
double[] expectedOutputs = new double[1];
long startTime = System.nanoTime();
long epochStartTime;
double avEpochTime;
int epochs = 0;
Random random = new Random();

// Loop until loss is acceptable.
while (loss > 0.01) {
    // Loop for sessionLength epochs and get the average loss across them.
    loss = 0;
    avEpochTime = 0;
    for (int i = 0; i < sessionLength; i++) {
        epochStartTime = System.nanoTime();
        // Create new inputs to XOR.
        inputs = new double[] {random.nextInt(2), random.nextInt(2)};

        // Determine what the inputs should return.
        if (inputs[0] == 1 && inputs[1] == 1)
            expectedOutputs[0] = 1.0;
        else
            expectedOutputs[0] = 0.0;

        // Perform a backPropagation.
        nn.backProp(inputs, expectedOutputs);
        // Sum our loss.
        loss += Math.abs(nn.getOutputs()[0] - expectedOutputs[0]) / sessionLength;
        avEpochTime += (System.nanoTime() - epochStartTime) / sessionLength;
    }
    // Increment epochs to keep track of how many have been performed.
    epochs += sessionLength;
    // Print a debugging statement.
    System.out.printf("Epochs: %5d | Loss: %6.4f | Average Time per Epoch: %6.0fns or %.4fms\n",
            epochs, loss, avEpochTime, avEpochTime / 1000000.0);
}
// Perform a few example XOR statements.
for (int i = 0; i < 10; i++) {
    // Generate new inputs.
    inputs = new double[] {random.nextInt(2), random.nextInt(2)};
    // Set our new inputs.
    nn.setInputs(inputs);
    // Forward propagate.
    nn.calculate();
    // Collect the outputs.
    outputs = nn.getOutputs();
    // Print out the XOR.
    System.out.printf("XOR Operation: %.0f XOR %.0f == %3.2f\n", inputs[0], inputs[1], outputs[0]);
}
// Print out our time to complete the training process.
double diffTime = System.nanoTime() - startTime;
System.out.printf("\nTime to complete for total training and example set: %.0fns or %.2fms\n", diffTime, diffTime / 1000000.0);
```
Note: If the loss does not converge, changing the learning rate and momentum can help.

# Saving to a File
Calling the saveToFile() method will store all of the relevant weight and bias data to a file with the name of a given string.This also does not keep any momentum stored before saving.
```java
nn.saveToFile("nn.txt");
```

# Loading from a File
Calling the loadFromFile() method loads a saved neural network to object making the call. It must be noted that you should initialize a neural network before loading the weights and biases from a file. This also does not keep any momentum stored before saving.
```java 
nn.loadFromFile("nn.txt");
```

# Thanks!
Thank you for looking at this little project. Feel free to make branches and recommendations for improvement as you see fit.
