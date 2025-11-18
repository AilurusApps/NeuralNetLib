namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Factory responsible for constructing <see cref="NeuralNetwork"/> instances.
    /// Encapsulates the creation and wiring of neurons, layers and connections so neural network construction
    /// logic is kept separate from runtime behavior.
    /// </summary>
    public static class NeuralNetworkFactory
    {
        /// <summary>
        /// Build a new neural network with the given topology.
        /// </summary>
        /// <param name="inputCount">The number of input neurons.</param>
        /// <param name="outputCount">The number of output neurons.</param>
        /// <param name="hiddenLayerCounts">An array describing the number of neurons in each hidden layer in forward order.
        /// <param name="activationFunction">Activation function to assign to neurons in the input and hidden layers. If unspecified, HyperTan will be used.</param>
        /// <param name="outputLayerFunction">Activation function to assign to neurons in the output layer. If unspecified, Sigmoid will be used.</param>
        /// <param name="weightInitializationStrategy">Weight initialization strategy to use for initializing weights. If unspecified, Xavier Normal initialization will be used.</param>
        /// Pass an empty array for no hidden layers.</param>
        /// <returns>A fully constructed <see cref="INeuralNetwork"/> ready for use.</returns>
        public static INeuralNetwork Build(
            int inputCount, 
            int outputCount, 
            int[] hiddenLayerCounts, 
            IActivationFunction? activationFunction = null, 
            IActivationFunction? outputLayerFunction = null, 
            IWeightInitializationStrategy? weightInitializationStrategy = null)
        {
            var inputs = CreateNeurons(inputCount, activationFunction ?? HyperTanFunction.Instance);

            var hiddenLayers = CreateHiddenLayers(inputs, hiddenLayerCounts, activationFunction ?? HyperTanFunction.Instance, 
                weightInitializationStrategy ?? XavierNormalInitalization.Instance).ToArray();

            var outputs = CreateAndConnectLayer(outputCount, hiddenLayers.LastOrDefault() ?? inputs, outputLayerFunction ?? SigmoidFunction.Instance,
                weightInitializationStrategy ?? XavierNormalInitalization.Instance);

            return new NeuralNetwork(inputs, hiddenLayers, outputs);
        }

        /// <summary>
        /// Create hidden layers for a network, connecting each new layer to the previous layer.
        /// </summary>
        /// <param name="inputs">The input layer to connect the first hidden layer to (if any).</param>
        /// <param name="hiddenLayerCounts">An array of node counts for each hidden layer.</param>
        /// <param name="activationFunction">Activation function to assign to neurons in the hidden layers.</param>
        /// <param name="weightInitializationStrategy">Weight initialization strategy to use for initializing weights.</param>
        /// <returns>An enumerable of hidden layer neuron arrays in forward order.</returns>
        private static IEnumerable<INeuron[]> CreateHiddenLayers(
            INeuron[] inputs, 
            int[] hiddenLayerCounts, 
            IActivationFunction activationFunction,
            IWeightInitializationStrategy weightInitializationStrategy)
        {
            var previousLayer = inputs;

            foreach (var layerNodeCount in hiddenLayerCounts)
            {
                var currentLayer = CreateAndConnectLayer(layerNodeCount, previousLayer, activationFunction, weightInitializationStrategy);
                previousLayer = currentLayer;
                yield return currentLayer;
            }
        }

        /// <summary>
        /// Create a layer of neurons and wire connections from the previous layer into the newly created layer.
        /// </summary>
        /// <param name="layerNodeCount">Number of neurons to create for this layer.</param>
        /// <param name="previousLayer">The layer that should be connected to the new layer as inputs.</param>
        /// <param name="activationFunction">Activation function to assign to neurons in this layer.</param>
        /// <param name="weightInitializationStrategy">Weight initialization strategy to use for initializing weights.</param>
        /// <returns>The newly created and connected neuron array for the layer.</returns>
        private static INeuron[] CreateAndConnectLayer(
            int layerNodeCount, 
            INeuron[] previousLayer, 
            IActivationFunction activationFunction, 
            IWeightInitializationStrategy weightInitializationStrategy)
        {
            var currentLayer = CreateNeurons(layerNodeCount, activationFunction);

            SetupConnections(previousLayer, currentLayer, weightInitializationStrategy);
            return currentLayer;
        }

        /// <summary>
        /// Create an array of neurons initialized with the provided activation function and a random bias.
        /// </summary>
        /// <param name="count">The number of neurons to create.</param>
        /// <param name="activationFunction">The activation function for each neuron.</param>
        /// <returns>An array of newly created neurons.</returns>
        private static INeuron[] CreateNeurons(int count, IActivationFunction activationFunction)
        {
            return Enumerable
                .Range(1, count)
                .Select(x => new Neuron(activationFunction, 0.01))
                .ToArray();
        }

        /// <summary>
        /// Wire connections from each neuron in <paramref name="currentLayer"/> to every neuron in <paramref name="nextLayer"/>.
        /// Existing outputs on the current layer will be replaced.
        /// </summary>
        /// <param name="currentLayer">The layer providing outputs (source neurons).</param>
        /// <param name="nextLayer">The layer receiving inputs (target neurons).</param>
        /// <param name="weightInitializationStrategy">Weight initialization strategy to use for initializing weights.</param>
        private static void SetupConnections(INeuron[] currentLayer, INeuron[] nextLayer, IWeightInitializationStrategy weightInitializationStrategy)
        {
            for (var i = 0; i < currentLayer.Length; i++)
            {
                var node = currentLayer[i];
                node.Outputs = CreateOutputConnections(node, nextLayer, i, currentLayer.Length, weightInitializationStrategy);
            }
        }

        /// <summary>
        /// Create connection objects from the given input neuron to every neuron in the output layer.
        /// Each created connection is also added to the corresponding index in the target neuron's <see cref="INeuron.Inputs"/> array.
        /// </summary>
        /// <param name="inputNode">The source neuron for the connections.</param>
        /// <param name="outputNodes">The target neurons to connect to.</param>
        /// <param name="currentNodeIndex">Index of the source neuron within its layer used when populating target input arrays.</param>
        /// <param name="currentLayerNodeCount">The number of neurons in the source layer; used to size target input arrays.</param>
        /// <param name="weightInitializationStrategy">Weight initialization strategy to use for initializing weights.</param>
        /// <returns>An array of created connections from the source neuron to each target neuron.</returns>
        private static IConnection[] CreateOutputConnections(
            INeuron inputNode, 
            INeuron[] outputNodes, 
            int currentNodeIndex, 
            int currentLayerNodeCount, 
            IWeightInitializationStrategy weightInitializationStrategy)
        {
            return outputNodes.Select(outputNode =>
            {
                var initialWeight = weightInitializationStrategy.GetInitialWeight(currentLayerNodeCount, outputNodes.Length);
                var connection = CreateConnection(inputNode, outputNode, initialWeight);

                AddConnectionToIndex(currentNodeIndex, currentLayerNodeCount, outputNode, connection);

                return connection;
            }).ToArray();
        }

        /// <summary>
        /// Add a connection to the target neuron's <see cref="INeuron.Inputs"/> array at the specified index.
        /// If the target's inputs array is not initialized it will be created with the provided size.
        /// </summary>
        /// <param name="index">Index at which to insert the connection.</param>
        /// <param name="currentLayerNodeCount">Size to allocate for the inputs array if it does not exist.</param>
        /// <param name="outputNode">The neuron whose inputs array will be updated.</param>
        /// <param name="connection">The connection to insert.</param>
        private static void AddConnectionToIndex(int index, int currentLayerNodeCount, INeuron outputNode, IConnection connection)
        {
            if (outputNode.Inputs == null)
                outputNode.Inputs = new IConnection[currentLayerNodeCount];

            outputNode.Inputs[index] = connection;
        }

        /// <summary>
        /// Create a single connection object between an input neuron and an output neuron with a random initial weight.
        /// </summary>
        /// <param name="inputNode">Source neuron for the connection.</param>
        /// <param name="outputNode">Target neuron for the connection.</param>
        /// <param name="initialWeight">The initial weight to use for the connection.</param>
        /// <returns>The constructed <see cref="IConnection"/> instance.</returns>
        private static IConnection CreateConnection(INeuron inputNode, INeuron outputNode, double initialWeight)
        {
            //return new Connection(inputNode, Random.Shared.NextDouble(), outputNode);
            //return new Connection(inputNode, -1 + (Random.Shared.NextDouble() * 0.2), outputNode);
            return new Connection(inputNode, initialWeight, outputNode);
        }
    }
}
