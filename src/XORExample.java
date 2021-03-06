import java.util.Random;

public class XORExample {
    public static void main(String[] args) throws Exception {
        // Create and initialize the NeuralNetwork.
        NeuralNetwork nn = new NeuralNetwork(2, 1, 1, new int[] {4}, 0.02);
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
        while (loss > 0.05) {
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
                loss += Math.pow(nn.getOutputs()[0] - expectedOutputs[0], 2) / sessionLength;
                avEpochTime += (System.nanoTime() - epochStartTime) / sessionLength;
            }
            // Increment epochs to keep track of how many have been performed.
            epochs += sessionLength;
            // Print a debugging statement.
            System.out.printf("Epochs: %5d | MSE: %6.4f | Average Time per Epoch: %6.0fns or %.4fms\n",
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
    }
}
