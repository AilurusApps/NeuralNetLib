namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents a feed-forward neural network composed of input neurons, zero or more hidden layers, and output neurons.
    /// The class is responsible for executing forward passes through the network (firing neurons) but delegates construction
    /// to a factory or other creator.
    /// </summary>
    public class NeuralNetwork : INeuralNetwork
    {
        /// <summary>
        /// The input neurons of the network. Each input neuron holds its current value and outgoing connections.
        /// </summary>
        public INeuron[] Inputs { get; private set; }

        /// <summary>
        /// The output neurons of the network. Outputs expose computed values after a forward pass and hold biases and gradients.
        /// </summary>
        public INeuron[] Outputs { get; private set; }

        /// <summary>
        /// The hidden layers of the network. Each element is an array of neurons representing one hidden layer in forward order.
        /// </summary>
        public INeuron[][] HiddenLayers { get; private set; }

        /// <summary>
        /// Internal constructor used by factory code to create a fully constructed network instance.
        /// Use a factory or public constructors to create instances in production code.
        /// </summary>
        /// <param name="inputs">Array of input neurons.</param>
        /// <param name="hiddenLayers">Array of hidden layer arrays (forward order).</param>
        /// <param name="outputs">Array of output neurons.</param>
        internal NeuralNetwork(INeuron[] inputs, INeuron[][] hiddenLayers, INeuron[] outputs)
        {
            Inputs = inputs;
            HiddenLayers = hiddenLayers;
            Outputs = outputs;
        }

        /// <summary>
        /// Performs a forward pass by assigning values to input neurons, firing them, and then propagating activation forward through the network.
        /// The number of provided input values must match the number of input neurons.
        /// </summary>
        /// <param name="inputValues">The input values to feed into the network.</param>
        public void Fire(params double[] inputValues)
        {
            for (var i = 0; i < inputValues.Length; i++)
            {
                var input = Inputs[i];
                input.Value = inputValues[i];
                input.Fire();
            }

            FeedForward();
        }

        /// <summary>
        /// Executes the forward propagation step for all hidden layers and output neurons.
        /// This method assumes input neurons have already been assigned their values and fired.
        /// </summary>
        public void FeedForward()
        {
            foreach (var hiddenLayer in HiddenLayers)
            {
                foreach (var node in hiddenLayer)
                {
                    node.Fire();
                }
            }

            foreach (var output in Outputs)
            {
                output.Fire();
            }
        }
    }
}
