namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents a training algorithm that can be applied to a neural network.
    /// Implementations perform learning by adjusting network weights and biases based on training examples.
    /// </summary>
    public interface ITrainingAlgorithm
    {
        /// <summary>
        /// Train the provided network using the supplied input values and expected output values.
        /// Implementations typically perform a forward pass followed by a parameter update (e.g., backpropagation).
        /// </summary>
        /// <param name="network">The neural network to train.</param>
        /// <param name="inputValues">The input values to feed into the network for this training example.</param>
        /// <param name="reward">The positive or negative reward signal.</param>
        /// <param name="expectedOutputValues">The expected target values for the network outputs.</param>
        public void Train(INeuralNetwork network, double[] inputValues, double reward, params double[] expectedOutputValues);
    }
}
