namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Implements the Rectified Linear Unit (ReLU) activation function and its derivative.
    /// This class exposes a singleton instance via the <see cref="Instance"/> property.
    /// </summary>
    public class ReLUFunction : IActivationFunction
    {
        private static readonly ReLUFunction _reluFunction = new();

        /// <summary>
        /// Singleton instance of <see cref="ReLUFunction"/> to be reused across the library.
        /// </summary>
        public static ReLUFunction Instance => _reluFunction;

        /// <summary>
        /// Compute the derivative of the ReLU function.
        /// Note: this method expects <paramref name="value"/> to be the activation output (i.e. ReLU(x)).
        /// For ReLU, the derivative is 1 if value > 0, otherwise 0.
        /// </summary>
        /// <param name="value">The activation output value (result of <see cref="Invoke(double)"/>).</param>
        /// <returns>The derivative of the ReLU function evaluated at the provided activation output.</returns>
        public double GetDerivativeValue(double value)
        {
            return value > 0 ? 1.0 : 0.0;
        }

        /// <summary>
        /// Compute the ReLU activation for the given input value.
        /// </summary>
        /// <param name="value">The input value (pre-activation, typically the weighted sum + bias).</param>
        /// <returns>The ReLU of the input value (max(0, value)).</returns>
        public double Invoke(double value)
        {
            return Math.Max(0.0, value);
        }
    }
}