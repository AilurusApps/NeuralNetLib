namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Implements the logistic sigmoid activation function and its derivative.
    /// This class exposes a singleton instance via the <see cref="Instance"/> property.
    /// </summary>
    public class SigmoidFunction : IActivationFunction
    {
        private static readonly SigmoidFunction _sigmoidFunction = new();

        /// <summary>
        /// Singleton instance of <see cref="SigmoidFunction"/> to be reused across the library.
        /// </summary>
        public static SigmoidFunction Instance => _sigmoidFunction;

        /// <summary>
        /// Compute the derivative of the sigmoid function.
        /// Note: this method expects <paramref name="value"/> to be the activation output (i.e. sigmoid(x)).
        /// For sigmoid, the derivative can be computed as value * (1 - value).
        /// </summary>
        /// <param name="value">The activation output value (result of <see cref="Invoke(double)"/>).</param>
        /// <returns>The derivative of the sigmoid function evaluated at the provided activation output.</returns>
        public double GetDerivativeValue(double value)
        {
            return value * (1 - value);
        }

        /// <summary>
        /// Compute the sigmoid activation for the given input value.
        /// </summary>
        /// <param name="value">The input value (pre-activation, typically the weighted sum + bias).</param>
        /// <returns>The sigmoid of the input value.</returns>
        public double Invoke(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }
    }
}
