namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents a neuron in the neural network. A neuron holds an activation function, bias, current value,
    /// gradient and references to incoming and outgoing connections. Implementations are expected to
    /// perform activation when <see cref="Fire"/> is called and to provide the activation derivative via <see cref="GetDerivativeValue"/>.
    /// </summary>
    public interface INeuron
    {
        /// <summary>
        /// The activation function used by the neuron to compute its output from a weighted sum.
        /// </summary>
        IActivationFunction ActivationFunction { get; set; }

        /// <summary>
        /// The neuron's bias term added to the weighted sum before applying the activation function.
        /// </summary>
        double Bias { get; set; }

        /// <summary>
        /// The computed gradient value used during backpropagation.
        /// </summary>
        double Gradient { get; set; }

        /// <summary>
        /// Incoming connections to this neuron. Can be null for input layer neurons.
        /// </summary>
        IConnection[]? Inputs { get; set; }

        /// <summary>
        /// Outgoing connections from this neuron. Can be null for output layer neurons.
        /// </summary>
        IConnection[]? Outputs { get; set; }

        /// <summary>
        /// The most recent change applied to this neuron's bias; used to support momentum in training algorithms.
        /// </summary>
        double PreviousBiasDelta { get; set; }

        /// <summary>
        /// The current value of the neuron (activation output). For input neurons this is typically set directly
        /// before propagation; for other neurons it is computed when <see cref="Fire"/> is called.
        /// </summary>
        double Value { get; set; }

        /// <summary>
        /// Compute the neuron's output by applying the activation function to the weighted sum of inputs plus bias.
        /// Implementations should update <see cref="Value"/> with the computed activation result.
        /// </summary>
        void Fire();

        /// <summary>
        /// Get the derivative of the activation function for the neuron's current output value.
        /// This is used during backpropagation to compute gradients. Implementations typically compute
        /// this by calling <see cref="IActivationFunction.GetDerivativeValue"/> with the neuron's current <see cref="Value"/>.
        /// </summary>
        /// <returns>The derivative of the activation function evaluated at the neuron's current output.</returns>
        double GetDerivativeValue();
    }
}