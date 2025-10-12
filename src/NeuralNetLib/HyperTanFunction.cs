namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Provides the hyperbolic tangent activation function and its derivative.
    /// This implementation exposes a singleton instance via the <see cref="Instance"/> property.
    /// </summary>
    public class HyperTanFunction : IActivationFunction
    {
        private static readonly HyperTanFunction _hyperTanFunction = new();

        /// <summary>
        /// Singleton instance of <see cref="HyperTanFunction"/> for reuse across the library.
        /// </summary>
        public static HyperTanFunction Instance => _hyperTanFunction;

        /// <summary>
        /// Computes the derivative of the tanh activation function.
        /// Note: this method expects the <paramref name="value"/> parameter to be the activation output (i.e. tanh(x)).
        /// The derivative of tanh(x) can be computed as 1 - tanh(x)^2; this implementation returns (1 - value) * (1 + value)
        /// which is mathematically equivalent to 1 - value^2.
        /// </summary>
        /// <param name="value">The activation output value (tanh(x)).</param>
        /// <returns>The derivative value at the provided activation output.</returns>
        public double GetDerivativeValue(double value)
        {
            return (1 - value) * (1 + value);
        }

        /// <summary>
        /// Invoke the hyperbolic tangent function on the provided input value.
        /// </summary>
        /// <param name="value">The input value (pre-activation).</param>
        /// <returns>The tanh of the input.</returns>
        public double Invoke(double value)
        {
            return Math.Tanh(value);
        }
    }
}
