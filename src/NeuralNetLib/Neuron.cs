namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents a neuron within the neural network. A neuron holds an activation function, bias,
    /// incoming and outgoing connections, and provides methods to compute its activation and derivative.
    /// </summary>
    /// <param name="activationFunction">The activation function used by the neuron.</param>
    /// <param name="initialBias">The initial bias value for the neuron.</param>
    public class Neuron(IActivationFunction activationFunction, double initialBias) : INeuron
    {
        /// <summary>
        /// Incoming connections to this neuron. Can be null for input layer neurons.
        /// </summary>
        public IConnection[]? Inputs { get; set; }

        /// <summary>
        /// Outgoing connections from this neuron. Can be null for output layer neurons.
        /// </summary>
        public IConnection[]? Outputs { get; set; }

        /// <summary>
        /// The current activation value of the neuron. For non-input neurons this value is computed by <see cref="Fire"/>.
        /// For input neurons this is typically set externally before propagation.
        /// </summary>
        public double Value { get; set; }

        /// <summary>
        /// The activation function used by the neuron to compute its output from the weighted sum of inputs plus bias.
        /// </summary>
        public IActivationFunction ActivationFunction { get; set; } = activationFunction;

        /// <summary>
        /// The gradient computed during backpropagation for this neuron.
        /// </summary>
        public double Gradient { get; set; }

        /// <summary>
        /// The neuron's bias term added to the weighted sum before applying the activation function.
        /// </summary>
        public double Bias { get; set; } = initialBias;

        /// <summary>
        /// The most recent change applied to this neuron's bias; used to support momentum-based training.
        /// </summary>
        public double PreviousBiasDelta { get; set; }

        /// <summary>
        /// Compute the neuron's activation value by applying the activation function to the weighted sum of inputs plus bias.
        /// If <see cref="Inputs"/> is null this method returns immediately (used for input neurons where value is set externally).
        /// The computed activation is stored in <see cref="Value"/>.
        /// </summary>
        public void Fire()
        {
            if (Inputs == null)
                return;

            Value = ActivationFunction.Invoke(Inputs.Sum(i => i.Weight * i.InputNode.Value) + Bias);
        }

        /// <summary>
        /// Returns the derivative of the activation function for the neuron's current activation value.
        /// Implementations typically call <see cref="IActivationFunction.GetDerivativeValue"/> with the neuron's current <see cref="Value"/>.
        /// </summary>
        /// <returns>The derivative of the activation function evaluated at the neuron's current output value.</returns>
        public double GetDerivativeValue()
        {
            return ActivationFunction.GetDerivativeValue(Value);
        }
    }
}
