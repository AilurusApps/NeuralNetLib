using System.Text;

namespace AilurusApps.NeuralNetLib.Serialization
{

    /// <summary>
    /// Provides methods to serialize a neural network's structure and state to a text-based stream or file.
    /// The serializer writes node counts, hidden layer lengths, biases, previous bias deltas, weights, and previous weight deltas
    /// as comma-delimited lines in a deterministic order.
    /// </summary>
    public static class NeuralNetworkSerializer
    {
        private const char _delimiter = ',';

        /// <summary>
        /// Serialize the provided neural network to the given stream using a simple comma-delimited text format.
        /// The stream is not closed by this method; a <see cref="StreamWriter"/> is created over the stream for the duration of the call.
        /// </summary>
        /// <param name="neuralNetwork">The network to serialize.</param>
        /// <param name="stream">The destination stream to write serialized data to.</param>
        public static void SerializeToStream(INeuralNetwork neuralNetwork, Stream stream)
        {
            using var writer = new StreamWriter(stream, Encoding.Default, leaveOpen: true);
            WriteNodeCounts(neuralNetwork, writer);
            WriteHiddenLayerLengths(neuralNetwork, writer);
            WriteBiases(neuralNetwork, writer);
            WritePreviousBiasDeltas(neuralNetwork, writer);
            WriteWeights(neuralNetwork, writer);
            WritePreviousWeightDeltas(neuralNetwork, writer);
        }

        /// <summary>
        /// Write a comma-delimited line containing all previous weight delta values for connections in the network.
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <param name="writer">The writer used to emit the line.</param>
        private static void WritePreviousWeightDeltas(INeuralNetwork neuralNetwork, StreamWriter writer)
        {
            writer.WriteLine(string.Join(_delimiter, GetPreviousWeightDeltaValues(neuralNetwork)));
        }

        /// <summary>
        /// Write a comma-delimited line containing all connection weights for the network.
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <param name="writer">The writer used to emit the line.</param>
        private static void WriteWeights(INeuralNetwork neuralNetwork, StreamWriter writer)
        {
            writer.WriteLine(string.Join(_delimiter, GetWeights(neuralNetwork)));
        }

        /// <summary>
        /// Write a comma-delimited line containing previous bias deltas for every neuron in the network (inputs, hidden, outputs).
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <param name="writer">The writer used to emit the line.</param>
        private static void WritePreviousBiasDeltas(INeuralNetwork neuralNetwork, StreamWriter writer)
        {
            writer.WriteLine(string.Join(_delimiter, GetPreviousBiasDeltaValues(neuralNetwork)));
        }

        /// <summary>
        /// Write a comma-delimited line containing bias values for every neuron in the network (inputs, hidden, outputs).
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <param name="writer">The writer used to emit the line.</param>
        private static void WriteBiases(INeuralNetwork neuralNetwork, StreamWriter writer)
        {
            writer.WriteLine(string.Join(_delimiter, GetBiasValues(neuralNetwork)));
        }

        /// <summary>
        /// Write a single line with the lengths of each hidden layer as comma-separated integers.
        /// </summary>
        /// <param name="neuralNetwork">The network whose hidden layer lengths will be written.</param>
        /// <param name="writer">The writer used to emit the line.</param>
        private static void WriteHiddenLayerLengths(INeuralNetwork neuralNetwork, StreamWriter writer)
        {
            writer.WriteLine(string.Join(_delimiter, neuralNetwork.HiddenLayers.Select(h => h.Length)));
        }

        /// <summary>
        /// Write a single line containing the input and output node counts (inputCount,outputCount).
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <param name="writer">The writer used to emit the line.</param>
        private static void WriteNodeCounts(INeuralNetwork neuralNetwork, StreamWriter writer)
        {
            writer.WriteLine(string.Join(_delimiter, neuralNetwork.Inputs.Length, neuralNetwork.Outputs.Length));
        }

        /// <summary>
        /// Enumerate all previous weight delta values from input-to-hidden and hidden-to-... connections in the network.
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <returns>A sequence of previous weight delta values in the same order they are written to the output.</returns>
        private static IEnumerable<double> GetPreviousWeightDeltaValues(INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.Inputs.SelectMany(i => i.Outputs!.Select(o => o.PreviousWeightDelta))
                .Concat(GetHiddenLayerPreviousWeightDeltas(neuralNetwork));
        }

        /// <summary>
        /// Enumerate previous weight deltas for connections that originate from hidden layers.
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <returns>A sequence of previous weight delta values for hidden-layer connections.</returns>
        private static IEnumerable<double> GetHiddenLayerPreviousWeightDeltas(INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.HiddenLayers.SelectMany(h => h.SelectMany(n => n.Outputs!.Select(o => o.PreviousWeightDelta)));
        }

        /// <summary>
        /// Enumerate all connection weights from inputs and hidden layers in the network.
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <returns>A sequence of weights.</returns>
        private static IEnumerable<double> GetWeights(INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.Inputs.SelectMany(i => i.Outputs!.Select(o => o.Weight))
                .Concat(GetHiddenLayerWeights(neuralNetwork));
        }

        /// <summary>
        /// Enumerate the weights for connections that originate from hidden layers.
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <returns>A sequence of hidden-layer connection weights.</returns>
        private static IEnumerable<double> GetHiddenLayerWeights(INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.HiddenLayers.SelectMany(h => h.SelectMany(n => n.Outputs!.Select(o => o.Weight)));
        }

        /// <summary>
        /// Enumerate previous bias delta values for all neurons in input, hidden and output layers.
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <returns>Sequence of previous bias delta values.</returns>
        private static IEnumerable<double> GetPreviousBiasDeltaValues(INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.Inputs.Select(i => i.PreviousBiasDelta)
                .Concat(neuralNetwork.HiddenLayers.SelectMany(h => h.Select(i => i.PreviousBiasDelta)))
                .Concat(neuralNetwork.Outputs.Select(o => o.PreviousBiasDelta));
        }

        /// <summary>
        /// Enumerate bias values for all neurons in input, hidden and output layers.
        /// </summary>
        /// <param name="neuralNetwork">The source network.</param>
        /// <returns>Sequence of bias values.</returns>
        private static IEnumerable<double> GetBiasValues(INeuralNetwork neuralNetwork)
        {
            return neuralNetwork.Inputs.Select(i => i.Bias)
                .Concat(neuralNetwork.HiddenLayers.SelectMany(h => h.Select(i => i.Bias)))
                .Concat(neuralNetwork.Outputs.Select(o => o.Bias));
        }

        /// <summary>
        /// Serialize the provided network to a file path. The file will be opened for writing and overwritten if it exists.
        /// </summary>
        /// <param name="neuralNetwork">The network to serialize.</param>
        /// <param name="filePath">The target file path to write serialized content into.</param>
        public static void SerializeToFile(INeuralNetwork neuralNetwork, string filePath)
        {
            using var file = File.OpenWrite(filePath);
            SerializeToStream(neuralNetwork, file);
        }
    }
}
