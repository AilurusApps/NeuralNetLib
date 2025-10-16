namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents a single training example consisting of input values and expected output values.
    /// This simple DTO is used by training algorithms to provide examples for learning.
    /// </summary>
    /// <param name="inputs">An array of input values for the training example.</param>
    /// <param name="outputs">One or more expected output values for the training example.</param>
    public class TrainingData(double[] inputs, params double[] outputs)
    {
        /// <summary>
        /// The input vector for this training example.
        /// </summary>
        public double[] Inputs { get; private set; } = inputs;

        /// <summary>
        /// The expected output vector for this training example.
        /// </summary>
        public double[] Outputs { get; private set; } = outputs;

        /// <summary>
        /// The positive or negative reward signal.
        /// </summary>
        public double? Reward { get; set; }
    }
}
