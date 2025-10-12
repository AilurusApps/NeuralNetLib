namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents a feed-forward neural network composed of input neurons, zero or more hidden layers, and output neurons.
    /// Provides operations to perform forward propagation through the network.
    /// </summary>
    public interface INeuralNetwork
    {
        /// <summary>
        /// The hidden layers of the network. Each element in the outer array is a layer consisting of neurons in forward order.
        /// </summary>
        INeuron[][] HiddenLayers { get; }

        /// <summary>
        /// The input neurons of the network. Input neurons receive external values which are then propagated forward.
        /// </summary>
        INeuron[] Inputs { get; }

        /// <summary>
        /// The output neurons of the network. After a forward pass these neurons expose the network's output values.
        /// </summary>
        INeuron[] Outputs { get; }

        /// <summary>
        /// Execute a forward propagation step for hidden layers and outputs. Assumes input neurons have been assigned their values and fired.
        /// </summary>
        void FeedForward();

        /// <summary>
        /// Assign values to input neurons, fire them, and perform a full forward pass.
        /// </summary>
        /// <param name="inputValues">Input values to feed into the network. The length must match the number of input neurons.</param>
        void Fire(params double[] inputValues);
    }
}