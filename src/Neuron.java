import java.util.Arrays;
import java.util.Random;

public class Neuron {
    // The next layer of neurons.
    private Neuron[] lastLayer;
    // The weights to the next layer of neurons.
    public double[] weights;
    // The weight for a 1.0 bias, we'll just add this to out sum when we call calculate().
    public double biasWeight;
    // The value of the current neuron.
    public double value;
    // The activation function being used.
    public int activationFunction;
    // The error for the neuron.
    public double error;
    // The derived value.
    public double derivative;
    // The previous weight change.
    public double[] prevChange;
    public double prevBiasChange;

    /**
     * Initializes the Neuron.
     * @param activationFunction The activation function to be used.
     *                           1 - Linear
     *                           2 - Sigmoid
     *                           3 - None
     */
    public Neuron(int activationFunction) {
        this.activationFunction = activationFunction;
        lastLayer = new Neuron[0];
    }

    /**
     * Pushes the current value to the next layer of neurons, according to the activation function of those connections.
     */
    public void calculate() {
        // Including out bias here, which is out biasWeight * 1.0.
        double sum = 0;
        for (int i = 0; i < lastLayer.length; i++) {
            sum += (lastLayer[i].value * weights[i]) + biasWeight;
        }
        value = doFunction(sum);
        derivative = getDerivative(value);
    }

    /**
     * Uses the relevant activation function on the given value.
     * @param sum The value to be altered.
     * @return The value which has had the relevant activation function applied.
     */
    public double getDerivative(double sum) {
        return switch (activationFunction) {
            case 1 -> linearDerive(sum);
            case 2 -> sigmoidDerive(sum);
            case 3 -> 1;
            default -> throw new IllegalStateException("Unexpected value: " + activationFunction);
        };
    }

    private double sigmoidDerive(double sum) {
        double x = 1 + Math.abs(sum);
        return 1 / (2 * (x * x));
    }

    private double linearDerive(double sum) {
        if (sum > 0) return sum;
        return 0;
    }

    /**
     * Uses the relevant activation function on the given value.
     * @param value The value to be altered.
     * @return The value which has had the relevant activation function applied.
     */
    public double doFunction(double value) {
        return switch (activationFunction) {
            case 1 -> linear(value);
            case 2 -> sigmoid(value);
            case 3 -> value;
            default -> throw new IllegalStateException("Unexpected value: " + activationFunction);
        };
    }

    /**
     * A linear ReLU function.
     * @param value The input value.
     * @return The output value, which will always be either <code>value</code> or 0.0.
     */
    private double linear(double value) {
        return Math.max(0.0, value);
    }

    /**
     * An efficient approximation of a sigmoid function, from -1.0 to 0.1, it also outputs negatives.
     * @param value The value to be altered.
     * @return Roughly equivalent to a sigmoid of <code>value</code>.
     */
    private double sigmoid(double value) {
        return 0.5 * (value / (1 + Math.abs(value))) + 0.5;
    }

    /**
     * Establishes random weights between this neuron and all neurons in <code>nextLayer</code>, [-4.0,4.0].
     */
    public void initialize() {
        weights = new double[lastLayer.length];
        prevChange = new double[lastLayer.length];
        Random random = new Random();
        for (int i = 0; i < lastLayer.length; i++) {
            weights[i] = (random.nextDouble() - 0.5) * 2.0;
        }
        biasWeight = (random.nextDouble() - 0.5) * 2.0;
        error = 0;
        derivative = 0;
        prevBiasChange = 0;
    }

    // Setters
    public void setNextLayer(Neuron[] nextLayer) {this.lastLayer = nextLayer;}
    public void setWeights(double[] weights) {this.weights = weights;}
    public void setActivationFunction(int activationFunction) {this.activationFunction = activationFunction;}
    // Getters
    public Neuron[] getNextLayer() {return lastLayer;}
    public double[] getWeights() {return weights;}
    public int getActivationFunction() {return activationFunction;}
}
