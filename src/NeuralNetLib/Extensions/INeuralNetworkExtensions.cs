namespace AilurusApps.NeuralNetLib.Extensions
{
    /// <summary>
    /// Extension methods for <see cref="INeuralNetwork"/> that provide convenient accessors
    /// for traversing neurons and connections across the whole network.
    /// </summary>
    public static class INeuralNetworkExtensions
    {
        /// <summary>
        /// Returns all outgoing connections from the input layer and all hidden layers.
        /// This does not include connections that originate from output neurons (outputs typically have no outputs).
        /// </summary>
        /// <param name="neuralNetwork">The neural network to enumerate connections for.</param>
        /// <returns>An enumerable of all output connections in the network.</returns>
        public static IEnumerable<IConnection> GetAllOutputConnections(this INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.Inputs.SelectMany(i => i.Outputs!).Concat(neuralNetwork.GetHiddenLayerOutputConnections());
        }

        /// <summary>
        /// Returns all outgoing connections from every neuron in every hidden layer.
        /// </summary>
        /// <param name="neuralNetwork">The neural network to enumerate hidden layer connections for.</param>
        /// <returns>An enumerable of connections originating from hidden layer neurons.</returns>
        public static IEnumerable<IConnection> GetHiddenLayerOutputConnections(this INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.HiddenLayers.SelectMany(h => h.SelectMany(n => n.Outputs!));
        }

        /// <summary>
        /// Returns every neuron in the network (inputs, hidden layers, and outputs) in forward order.
        /// </summary>
        /// <param name="neuralNetwork">The neural network to enumerate neurons for.</param>
        /// <returns>An enumerable of all neurons in the network.</returns>
        public static IEnumerable<INeuron> GetAllNodes(this INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.Inputs.Concat(neuralNetwork.GetAllHiddenLayerNodes()).Concat(neuralNetwork.Outputs);
        }

        /// <summary>
        /// Returns every neuron contained in all hidden layers in forward order.
        /// </summary>
        /// <param name="neuralNetwork">The neural network to enumerate hidden layer neurons for.</param>
        /// <returns>An enumerable of neurons from all hidden layers.</returns>
        private static IEnumerable<INeuron> GetAllHiddenLayerNodes(this INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.HiddenLayers.SelectMany(h => h);
        }
    }
}
