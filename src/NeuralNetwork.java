import java.io.*;

public class NeuralNetwork {
    public Neuron[][] neurons;
    public final int layers;
    public double learningRate;
    public double momentum;
    private double errorSum;

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

        // Used for mini-batching.
        errorSum = 0;

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
     * Copies the weights and biases from the given network to the current one.
     * Ensure both networks are of equal size.
     * @param parent The network whose weights are to be copied from.
     */
    public void copyWeightFrom(NeuralNetwork parent) {
        // Loops for each layer except the output layer.
        for (int layer = 1; layer < layers; layer++) {
            // Loops for each neuron on the current layer.
            for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
                neurons[layer][neuron].biasWeight = parent.neurons[layer][neuron].biasWeight;
                for (int weight = 0; weight < neurons[layer][neuron].weights.length; weight++) {
                    neurons[layer][neuron].weights[weight] = parent.neurons[layer][neuron].weights[weight];
                }
            }
        }
    }

    /**
     * A copy constructor for NeuralNetwork.
     * @param neuralNetwork The NeuralNetwork to be copied from.
     */
    public NeuralNetwork(NeuralNetwork neuralNetwork) {
        learningRate = neuralNetwork.learningRate;
        momentum = neuralNetwork.momentum;
        this.errorSum = neuralNetwork.errorSum;

        // Get the parent network to copy from.
        Neuron[][] parentNetwork = neuralNetwork.neurons;

        // Get the parent's number of layers.
        this.layers = parentNetwork.length;

        // Initialize the new network's layers.
        neurons = new Neuron[layers][];

        // Initialize the new network's inputs.
        neurons[0] = new Neuron[parentNetwork[0].length];
        for (int i = 0; i < neurons[0].length; i++) {
            neurons[0][i] = new Neuron(3);
        }

        // Loops for each layer except the output layer.
        for (int layer = 1; layer < layers; layer++) {
            neurons[layer] = new Neuron[parentNetwork[layer].length];
            // Loops for each neuron on the current layer.
            for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
                // Copies and initializes the neuron.
                neurons[layer][neuron] = new Neuron(parentNetwork[layer][neuron], neurons[layer - 1]);
            }
        }
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
                neurons[layer][neuron].setLastLayer(neurons[layer - 1]);
                neurons[layer][neuron].initialize();
            }
        }
    }

    /**
     * This function calculates the output(s) of the neural network, which are stored in the output layer. Call
     * getOutputs() to access this processed data.
     */
    public void calculate() {
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
    public void backProp(double[] inputs, double[] expectedOutput) throws Exception {
        // Checking that the given array is of equal size to the output layer.
        if (expectedOutput.length != neurons[neurons.length - 1].length) {
            throw new Exception("Error: Expected an array equal in size to the output layer, got " +
                    expectedOutput.length + " when expecting " + neurons[neurons.length - 1].length);
        }
        setInputs(inputs);

        // Push data through network.
        calculate();

        // Calculate input error.
        Neuron[] layer = neurons[neurons.length - 1];
        for (int i = 0; i < layer.length; i++) {
            layer[i].error = layer[i].derivative * (layer[i].value - expectedOutput[i]);
        }

        // Calculate hidden layer error.
        for (int i = neurons.length - 2; i > 0; i--) {
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
                neuron.prevBiasChange = (learningRate * neuron.error) + (momentum * temp);
                neuron.biasWeight -= neuron.prevBiasChange;
                for (int weight = 0; weight < neuron.weights.length; weight++) {
                    temp = neuron.prevChange[weight];
                    neuron.prevChange[weight] = (learningRate * neuron.error * neurons[i - 1][weight].value) + (momentum * temp);
                    neuron.weights[weight] -= neuron.prevChange[weight];
                }
            }
        }
    }

    /**
     * Use this method to sum the error for a given example. Use this across multiple examples to then use the
     * backPropSumError() method to update the neural network to those errors.
     * @param expectedOutput The values by which errors are summed from.
     * @throws Exception     Ensure that expectedOutput is equal in length to the output layer.
     */
    public void sumError(double[] input, double[] expectedOutput) throws Exception {
        // Checking that the given array is of equal size to the output layer.
        if (expectedOutput.length != neurons[neurons.length - 1].length) {
            throw new Exception("Error: Expected an array equal in size to the output layer, got " +
                    expectedOutput.length + " when expecting " + neurons[neurons.length - 1].length);
        }

        errorSum += 1.0;

        // Set our input data.
        setInputs(input);

        // Push data through network.
        calculate();

        // Calculate input error.
        Neuron[] layer = neurons[neurons.length - 1];
        for (int i = 0; i < layer.length; i++) {
            layer[i].error += layer[i].derivative * (layer[i].value - expectedOutput[i]);
        }

        // Calculate hidden layer error.
        for (int i = neurons.length - 2; i >= 0; i--) {
            layer = neurons[i];
            // Compute error of layer's neurons.
            for (int j = 0; j < layer.length; j++) {
                double error = 0.0;
                for (int neuron = 0; neuron < neurons[i + 1].length; neuron++) {
                    error = neurons[i + 1][neuron].weights[j] * neurons[i + 1][neuron].error;
                }
                layer[j].error += error * layer[j].derivative;
            }
        }
    }

    /**
     * This method performs a back propagation algorithm and resets the error value for each node.
     * Use this with the sumError() method to fit the model to a given set of examples.
     */
    public void backPropSumError() {
        if (errorSum == 0)
            errorSum = 1.0;
        // Update weights.
        double temp;
        Neuron[] layer;
        for (int i = layers - 1; i > 0; i--) {
            layer = neurons[i];
            for (Neuron neuron : layer) {
                // Updating our bias.
                temp = neuron.prevBiasChange;
                neuron.prevBiasChange = ((learningRate * neuron.error) + (momentum * temp) ) / errorSum;
                neuron.biasWeight -= neuron.prevBiasChange;
                for (int weight = 0; weight < neuron.weights.length; weight++) {
                    temp = neuron.prevChange[weight];
                    neuron.prevChange[weight] = ((learningRate * neuron.error * neurons[i - 1][weight].value) + (momentum * temp)) / errorSum;
                    neuron.weights[weight] -= neuron.prevChange[weight];
                }
                neuron.error = 0.0;
            }
        }
        errorSum = 0;
    }

    /**
     * Sets a layer of neurons to have a specific activation function.
     * @param layer              The index of the layer to have its activation function changed.
     * @param activationFunction The new activation function.
     *                           1 - Linear (ReLU)
     *                           2 - Sigmoid
*                                3 - Sum
     */
    public void setLayerActivation(int layer, int activationFunction) {
        for (int i = 0; i < neurons[layer].length; i++)
            neurons[layer][i].setActivationFunction(activationFunction);
    }

    /**
     * Save the neural network to a txt file.
     * @param fileName     The name of the File to be writen to. Use .txt at the end of the given string.
     * @throws IOException Thrown if the BufferedWriter encounters an error.
     */
    public void saveToFile(String fileName) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
        StringBuilder data;
        for (int layer = 0; layer < neurons.length; layer++) {
            data = new StringBuilder();
            for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
                data.append(neurons[layer][neuron].activationFunction).append(",").append(neurons[layer][neuron].biasWeight);
                if (layer > 0) {
                    for (int weight = 0; weight < neurons[layer][neuron].weights.length; weight++) {
                        data.append(",").append(neurons[layer][neuron].weights[weight]);
                    }
                }
                data.append("|");
            }
            bw.write(data.toString());
            bw.newLine();
        }
        bw.close();

    }

    /**
     * Loads the given file onto the neural network. This will completely overwrite the current neural network with
     * new weights, biases, and neurons - and remove any momentum saved from before.
     * @param fileName     The name of the File to be read from. Use .txt at the end of the given string.
     * @throws IOException Thrown if the BufferedReader encounters an error.
     */
    public void loadFromFile(String fileName) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        String layer;
        String[] neuron;
        neurons = new Neuron[0][];
        Neuron[][] temp;
        int layerIndex = 0;
        while ((layer = br.readLine()) != null) {
            neuron = layer.split("\\|");
            temp = new Neuron[neurons.length][];
            for (int i = 0; i < neurons.length; i++) {
                temp[i] = new Neuron[neurons[i].length];
                for (int j = 0; j < neurons[i].length; j++) {
                    temp[i][j] = neurons[i][j];
                }
            }

            neurons = new Neuron[neurons.length + 1][];

            for (int i = 0; i < temp.length; i++) {
                neurons[i] = new Neuron[temp[i].length];
                for (int j = 0; j < temp[i].length; j++) {
                    neurons[i][j] = temp[i][j];
                }
            }

            neurons[layerIndex] = new Neuron[neuron.length];
            for (int i = 0; i < neuron.length; i++) {
                for (int cursor = 0; cursor < neuron[i].length(); cursor++) {

                    if (layerIndex == 0) {
                        neurons[layerIndex][i] = new Neuron(Integer.parseInt(neuron[i].substring(0, neuron[i].indexOf(","))));
                        neurons[layerIndex][i].biasWeight = Double.parseDouble(neuron[i].substring(neuron[i].indexOf(",") + 1));
                    }

                    else {
                        neurons[layerIndex][i] = new Neuron(Integer.parseInt(neuron[i].substring(0, neuron[i].indexOf(","))));
                        neuron[i] = neuron[i].substring(neuron[i].indexOf(",") + 1);
                        neurons[layerIndex][i].biasWeight = Double.parseDouble(neuron[i].substring(0, neuron[i].indexOf(",")));
                        neuron[i] = neuron[i].substring(neuron[i].indexOf(",") + 1);
                        String[] weights = neuron[i].split(",");
                        neurons[layerIndex][i].weights = new double[weights.length];
                        for (int weightIndex = 0; weightIndex < weights.length; weightIndex++) {
                            neurons[layerIndex][i].weights[weightIndex] = Double.parseDouble(weights[weightIndex]);
                        }
                        neuron[i] = "";
                        neurons[layerIndex][i].setLastLayer(neurons[layerIndex - 1]);
                        neurons[layerIndex][i].prevChange = new double[neurons[layerIndex - 1].length];
                        neurons[layerIndex][i].prevBiasChange = 0;
                        neurons[layerIndex][i].error = 0;
                    }
                }
            }

            layerIndex++;
        }
        br.close();
    }
}
