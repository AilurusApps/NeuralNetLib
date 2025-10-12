using System.Runtime.CompilerServices;

namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Implements the backpropagation training algorithm for a feed-forward neural network.
    /// Responsible for computing gradients and updating weights and biases using learning rate and momentum.
    /// </summary>
    public class Backpropagation : ITrainingAlgorithm
    {
        /// <summary>
        /// Learning rate (eta) used to scale weight and bias updates.
        /// Typical values are small (e.g. 0.01 - 0.1).
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Momentum coefficient used to smooth updates across iterations.
        /// Previous weight/bias deltas are scaled by this factor and added to the current update.
        /// </summary>
        public double Momentum { get; set; } 

        /// <summary>
        /// Train the provided network using the supplied input values and expected output values.
        /// This method performs a forward pass (Fire) followed by backpropagation to update weights and biases.
        /// </summary>
        /// <param name="network">The neural network to train.</param>
        /// <param name="inputValues">Input values to feed into the network.</param>
        /// <param name="expectedOutputValues">Expected target values for the network outputs.</param>
        public void Train(INeuralNetwork network, double[] inputValues, params double[] expectedOutputValues)
        {
            network.Fire(inputValues);

            Backpropagate(network, expectedOutputValues);
        }

        /// <summary>
        /// Execute the backpropagation routine on the given network using the expected output values.
        /// This computes gradients for output and hidden layers, then updates weights and biases.
        /// </summary>
        /// <param name="network">The neural network to update.</param>
        /// <param name="expectedOutputValues">Expected target values for the network outputs.</param>
        public void Backpropagate(INeuralNetwork network, params double[] expectedOutputValues)
        {
            UpdateOutputGradients(network, expectedOutputValues);

            UpdateHiddenLayerGradients(network, expectedOutputValues);

            UpdateInputWeights(network);

            UpdateHiddenLayerWeightsAndBiases(network);

            UpdateOutputBiases(network);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Update the biases for all output neurons using their computed gradients.
        /// </summary>
        /// <param name="network">The network containing output neurons.</param>
        private void UpdateOutputBiases(INeuralNetwork network)
        {
            foreach (var output in network.Outputs)
            {
                UpdateBias(output);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Update biases and outgoing weights for all hidden layer neurons.
        /// Iterates hidden layers in forward order (from input-side to output-side).
        /// </summary>
        /// <param name="network">The network containing hidden layers.</param>
        private void UpdateHiddenLayerWeightsAndBiases(INeuralNetwork network)
        {
            foreach (var layer in network.HiddenLayers)
            {
                foreach (var hiddenNode in layer)
                {
                    UpdateBias(hiddenNode);

                    UpdateOutputWeights(hiddenNode);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Update all outgoing connection weights for a hidden neuron.
        /// </summary>
        /// <param name="hiddenNode">The hidden neuron whose output connections should be updated.</param>
        private void UpdateOutputWeights(INeuron hiddenNode)
        {
            if (hiddenNode.Outputs == null)
                return;

            foreach (var connection in hiddenNode.Outputs)
            {
                UpdateWeight(connection);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Update all weights that originate from input neurons.
        /// </summary>
        /// <param name="network">The network containing input neurons.</param>
        private void UpdateInputWeights(INeuralNetwork network)
        {
            foreach (var input in network.Inputs)
            {
                if (input.Outputs == null)
                    continue;

                foreach (var connection in input.Outputs)
                {
                    UpdateWeight(connection);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Update a single connection's weight using the gradient of the output node,
        /// the input node value, the learning rate, and momentum.
        /// </summary>
        /// <param name="connection">The connection to update.</param>
        private void UpdateWeight(IConnection connection)
        {
            var delta = LearningRate * connection.OutputNode.Gradient * connection.InputNode.Value;
            connection.Weight += delta + (Momentum * connection.PreviousWeightDelta);
            connection.PreviousWeightDelta = delta;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Update a neuron's bias using its gradient, the learning rate, and momentum.
        /// </summary>
        /// <param name="node">The neuron whose bias will be updated.</param>
        private void UpdateBias(INeuron node)
        {
            var delta = LearningRate * node.Gradient;
            node.Bias += delta + (Momentum * node.PreviousBiasDelta);
            node.PreviousBiasDelta = delta;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Compute and assign gradients for each output neuron based on expected targets.
        /// </summary>
        /// <param name="network">The network containing output neurons.</param>
        /// <param name="expectedOutputValues">Target values for the outputs.</param>
        private static void UpdateOutputGradients(INeuralNetwork network, double[] expectedOutputValues)
        {
            for (var o = 0; o < expectedOutputValues.Length; o++)
            {
                var output = network.Outputs[o];
                output.Gradient = CalculateGradient(output, expectedOutputValues[o]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Calculate the gradient for a neuron given its target value.
        /// Uses the neuron's activation derivative and the error (target - value).
        /// </summary>
        /// <param name="node">The neuron for which to calculate the gradient.</param>
        /// <param name="targetValue">The expected target value for the neuron.</param>
        /// <returns>The computed gradient.</returns>
        private static double CalculateGradient(INeuron node, double targetValue)
        {
            return node.GetDerivativeValue() * (targetValue - node.Value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Compute gradients for hidden layers in reverse order (from last hidden layer to first).
        /// </summary>
        /// <param name="network">The network containing hidden layers.</param>
        /// <param name="expectedOutputValues">Target values for the outputs (not used directly but kept for signature consistency).</param>
        private static void UpdateHiddenLayerGradients(INeuralNetwork network, double[] expectedOutputValues)
        {
            for (var l = network.HiddenLayers.Length - 1; l >= 0; l--)
            {
                var layer = network.HiddenLayers[l];

                UpdateHiddenNodeGradients(layer);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Compute gradients for every node in a hidden layer using the gradients and weights
        /// of their output connections.
        /// </summary>
        /// <param name="layer">Array of neurons forming the hidden layer.</param>
        private static void UpdateHiddenNodeGradients(INeuron[] layer)
        {
            foreach (var hiddenNode in layer)
            {
                hiddenNode.Gradient = CalculateGradient(hiddenNode, hiddenNode.Outputs!);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Calculate the gradient for a hidden neuron given its output connections.
        /// The gradient is the derivative of the activation multiplied by the weighted sum of downstream gradients.
        /// </summary>
        /// <param name="node">The hidden neuron.</param>
        /// <param name="outputConnections">Connections from this neuron to the next layer.</param>
        /// <returns>The computed gradient value.</returns>
        private static double CalculateGradient(INeuron node, IConnection[] outputConnections)
        {
            return node.GetDerivativeValue() * outputConnections.Sum(o => o.OutputNode.Gradient * o.Weight);
        }
    }
}
