public class NeuralNetwork {
    public Neuron[][] neurons;
    public final int layers;
    public double fitness;
    public double learningRate;
    public double momentum;

    /**
     * @param inputs The number of inputs.
     * @param outputs The number of outputs.
     * @param hiddenLayers The number of hidden layers.
     * @param layerNeurons An integer array specifying the number of neurons in each hidden layer.
     */
    public NeuralNetwork(int inputs, int outputs, int hiddenLayers, int[] layerNeurons, double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        // Initialization of neurons.
        layers = 2 + hiddenLayers;
        neurons = new Neuron[layers][];
        fitness = 0;

        // Input layer initialization.
        neurons[0] = new Neuron[inputs];
        for (int i = 0; i < inputs; i++) {
            neurons[0][i] = new Neuron(3);
        }

        // Output layer initialization.
        neurons[layers - 1] = new Neuron[outputs];
        for (int i = 0; i < outputs; i++) {
            neurons[layers - 1][i] = new Neuron(2);
        }

        // Hidden layer initialization.
        for (int i = 1; i < layers - 1; i++) {
            neurons[i] = new Neuron[layerNeurons[i - 1]];
            for (int j = 0; j < layerNeurons[i - 1]; j++) {
                neurons[i][j] = new Neuron(1);
            }
        }
    }

    /**
     * A copy constructor for NeuralNetwork.
     * @param NeuralNetwork The NeuralNetwork to be copied from.
     */
    public NeuralNetwork(NeuralNetwork NeuralNetwork) {
        this.layers = NeuralNetwork.layers;
        initialize();
    }

    /**
     * Initializes and randomizes the connections between each layer in the neural network. This must be done before
     * data is input, or no output will be generated.
     */
    public void initialize() {
        // Loops for each layer except the output layer.
        for (int layer = 1; layer < layers; layer++) {
            // Loops for each neuron on the current layer.
            for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
                neurons[layer][neuron].setNextLayer(neurons[layer - 1]);
                neurons[layer][neuron].initialize();
            }
        }
    }

    /**
     * This function calculates the output(s) of the neural network, which are stored in the output layer. Call
     * getOutputs() to access this processed data.
     */
    public void calculate() {
        int lastLayer = neurons.length - 1;
        // Loops for each layer except the output layer.
        for (int layer = 1; layer < layers; layer++) {
            // Loops for each neuron on the current layer.
            for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
                neurons[layer][neuron].calculate();
            }
        }
    }

    /**
     * In order for these values returned to be valid to the dataset provided, use the calculate() function.
     * @return An array of doubles of the size of the output layer, which are the values of the corresponding neurons.
     */
    public double[] getOutputs() {
        double[] outputs = new double[neurons[neurons.length - 1].length];
        for (int i = 0; i < neurons[neurons.length - 1].length; i++)
            outputs[i] = neurons[neurons.length - 1][i].value;
        return outputs;
    }

    /**
     * Sets the input values of the neural network - must be an array of doubles of equal size to the input layer.
     * This must be done before calling @calculate() or the output will be useless.
     * @param inputs The array of doubles representing data.
     */
    public void setInputs(double[] inputs) {
        for (int i = 0; i < neurons[0].length; i++)
            neurons[0][i].value = inputs[i];
    }

    /**
     * This function performs backpropagation depending on the set input. Ensure an input is set before calling this
     * method.
     * @param expectedOutput The values which are expected from the neural network.
     */
    public void backProp(double[] expectedOutput) throws Exception {
        // Checking that the given array is of equal size to the output layer.
        if (expectedOutput.length != neurons[neurons.length - 1].length) {
            throw new Exception("Error: Expected an array equal in size to the output layer, got " +
                    expectedOutput.length + " when expecting " + neurons[neurons.length - 1].length);
        }

        // Push data through network.
        calculate();

        // Calculate input error.
        Neuron[] layer = neurons[neurons.length - 1];
        for (int i = 0; i < layer.length; i++) {
            layer[i].error = layer[i].derivative * (layer[i].value - expectedOutput[i]);
        }

        // Calculate hidden layer error.
        for (int i = layers - 2; i >= 0; i--) {
            layer = neurons[i];
            // Compute error of layer's neurons.
            for (int j = 0; j < layer.length; j++) {
                double error = 0.0;
                for (int neuron = 0; neuron < neurons[i + 1].length; neuron++) {
                    error += neurons[i + 1][neuron].weights[j] * neurons[i + 1][neuron].error;
                }
                layer[j].error = error * layer[j].derivative;
            }
        }

        // Update weights.
        double temp;
        for (int i = layers - 1; i > 0; i--) {
            layer = neurons[i];
            for (Neuron neuron : layer) {
                // Updating our bias.
                temp = neuron.prevBiasChange;
                neuron.prevBiasChange = learningRate * neuron.error;
                neuron.biasWeight -= neuron.prevBiasChange + (momentum * temp);
                for (int weight = 0; weight < neuron.weights.length; weight++) {
                    temp = neuron.prevChange[weight];
                    neuron.prevChange[weight] = learningRate * neuron.error * neurons[i - 1][weight].value;
                    neuron.weights[weight] -= neuron.prevChange[weight] + (momentum * temp);
                }
            }
        }
    }

    public void setFitness(double fitness) {this.fitness = fitness;}

    public void setLayerActivation(int layer, int activationFunction) {
        for (int i = 0; i < neurons[layer].length; i++)
            neurons[layer][i].setActivationFunction(activationFunction);
    }
}
