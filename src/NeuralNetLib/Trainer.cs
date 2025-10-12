using System.Runtime.CompilerServices;

namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Coordinates training of neural networks using a supplied training algorithm and a collection of training examples.
    /// The generic key <typeparamref name="TKey"/> identifies training examples stored in an internal dictionary.
    /// </summary>
    /// <typeparam name="TKey">Type used as the key for identifying training examples in the training data dictionary.</typeparam>
    /// <param name="trainingData">A dictionary of training examples keyed by <typeparamref name="TKey"/>.</param>
    /// <param name="algorithm">The training algorithm used to update network weights (e.g., backpropagation).</param>
    public class Trainer<TKey>(IDictionary<TKey, TrainingData> trainingData, ITrainingAlgorithm algorithm)
    {
        private readonly IDictionary<TKey, TrainingData> _trainingData = trainingData;
        private readonly ITrainingAlgorithm _algorithm = algorithm;

        /// <summary>
        /// Gets the collection of training data values managed by this trainer.
        /// </summary>
        public IEnumerable<TrainingData> TrainingData { get => _trainingData.Values; }

        /// <summary>
        /// Add a new training example to the internal collection or update an existing entry with the provided key.
        /// </summary>
        /// <param name="key">The key identifying the training example.</param>
        /// <param name="value">The training data to add or update.</param>
        public void AddOrUpdateData(TKey key, TrainingData value)
        {
            _trainingData[key] = value;
        }

        /// <summary>
        /// Train the provided neural network on a single training example until the output error falls below the specified tolerance
        /// or until the maximum number of iterations is reached.
        /// </summary>
        /// <param name="neuralNetwork">The neural network to train.</param>
        /// <param name="tolerance">The acceptable maximum error for the network outputs.</param>
        /// <param name="maxIterations">Maximum number of training iterations to perform.</param>
        /// <param name="data">The training example to use for training.</param>
        /// <returns>True if training achieved the tolerance within the iteration limit; otherwise false.</returns>
        public bool Train(INeuralNetwork neuralNetwork, double tolerance, int maxIterations, TrainingData data)
        {
            double maxError = 0;

            for (var i = 0; i < maxIterations; i++)
            {
                maxError = TrainAndGetMaxError(neuralNetwork, data, maxError);
                if (maxError <= tolerance)
                    return true;
            }

            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Perform a single training step using the configured algorithm and return the updated maximum output error.
        /// </summary>
        /// <param name="neuralNetwork">The network to train for this step.</param>
        /// <param name="data">The training example to use.</param>
        /// <param name="maxError">The current maximum error value before this step; the method returns the updated maximum.</param>
        /// <returns>The updated maximum output error after training on the provided example.</returns>
        private double TrainAndGetMaxError(INeuralNetwork neuralNetwork, TrainingData data, double maxError)
        {
            _algorithm.Train(neuralNetwork, data.Inputs, data.Outputs);
            maxError = Math.Max(maxError, GetMaxOutputError(neuralNetwork, data));
            return maxError;
        }

        /// <summary>
        /// Retrain the network using all stored training examples until the maximum output error is below the tolerance
        /// or until the maximum number of iterations (summed across examples) is reached.
        /// </summary>
        /// <param name="neuralNetwork">The network to retrain.</param>
        /// <param name="tolerance">The acceptable maximum output error.</param>
        /// <param name="maxIterations">Maximum total number of training iterations to perform across all examples.</param>
        /// <returns>True if retraining achieved the tolerance within the iteration limit; otherwise false.</returns>
        public bool Retrain(INeuralNetwork neuralNetwork, double tolerance, int maxIterations)
        {
            double maxError = 0;
            var iteration = 0;

            do
            {
                foreach (var data in _trainingData.Values)
                {
                    maxError = TrainAndGetMaxError(neuralNetwork, data, maxError);
                    if (iteration++ >= maxIterations)
                        break;
                }
            } while (iteration < maxIterations && maxError > tolerance);

            return maxError <= tolerance;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Compute the maximum absolute difference between the network outputs and the expected values for a single training example.
        /// </summary>
        /// <param name="neuralNetwork">The network whose outputs will be compared.</param>
        /// <param name="data">The expected outputs for comparison.</param>
        /// <returns>The maximum absolute error across all output neurons for the provided example.</returns>
        private static double GetMaxOutputError(INeuralNetwork neuralNetwork, TrainingData data)
        {
            return neuralNetwork.Outputs.Select((o, n) => Math.Abs(Math.Abs(o.Value) - Math.Abs(data.Outputs[n]))).Max();
        }
    }
}
