namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents an activation function used by neurons in the neural network.
    /// Implementations should provide the activation (Invoke) and a method to compute the derivative
    /// which is used during training (backpropagation).
    /// </summary>
    public interface IActivationFunction
    {
        /// <summary>
        /// Compute the activation function for the given input value (pre-activation).
        /// </summary>
        /// <param name="value">The input value (often the weighted sum delivered to the neuron).</param>
        /// <returns>The activated output value.</returns>
        double Invoke(double value);

        /// <summary>
        /// Compute the derivative of the activation function.
        /// Note: implementations in this project expect <paramref name="value"/> to be the activation output
        /// (i.e. <c>Invoke(x)</c>) rather than the pre-activation input. This is done for efficiency in backpropagation.
        /// </summary>
        /// <param name="value">The activation output value (result of <see cref="Invoke"/>).</param>
        /// <returns>The derivative value at the provided activation output.</returns>
        double GetDerivativeValue(double value);
    }
}
