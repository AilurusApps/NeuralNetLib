using System.Text;
using AilurusApps.NeuralNetLib.Extensions;

namespace AilurusApps.NeuralNetLib.Serialization
{
    /// <summary>
    /// Provides functionality to deserialize neural networks from CSV data in files and streams.
    /// The expected input format is a sequence of comma-delimited lines in the following order:
    /// 1) inputCount,outputCount
    /// 2) hiddenLayerLengths (comma separated lengths, or empty for none)
    /// 3) biases (one value per neuron: inputs, hidden layers in forward order, outputs)
    /// 4) previousBiasDeltas (matching the biases order)
    /// 5) weights (all connection weights in the order they are emitted by the serializer)
    /// 6) previousWeightDeltas (matching the weights order)
    /// </summary>
    public static class NeuralNetworkDeserializer
    {
        private const string _invalidHeaderMessage = "Invalid header line. Expected format: inputCount,outputCount";
        private const char _delimiter = ',';

        /// <summary>
        /// Deserializes a neural network from a file at the given path.
        /// </summary>
        /// <param name="filePath">The path to the file containing the neural network data.</param>
        /// <returns>A deserialized <see cref="INeuralNetwork"/> instance.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="filePath"/> is null.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the specified file is not found.</exception>
        /// <exception cref="InvalidDataException">Thrown when the file format is invalid.</exception>
        public static INeuralNetwork DeserializeFromFile(string filePath)
        {
            using var stream = File.OpenRead(filePath);

            return DeserializeFromStream(stream);
        }

        /// <summary>
        /// Deserializes a neural network from the provided stream using the expected CSV layout.
        /// </summary>
        /// <param name="stream">The stream containing the neural network data.</param>
        /// <returns>A deserialized <see cref="INeuralNetwork"/> instance.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="stream"/> is null.</exception>
        /// <exception cref="InvalidDataException">Thrown when the stream data format is invalid.</exception>
        public static INeuralNetwork DeserializeFromStream(Stream stream)
        {
            using var reader = new StreamReader(stream, Encoding.Default);

            string[]? line1 = ReadNodeCountLine(reader);
            var hiddenLayerCounts = ReadHiddenLayerCounts(reader);

            var neuralNetwork = NeuralNetworkFactory.Build(ReadAsInt(line1[0], "inputCount"), ReadAsInt(line1[1], "outputCount"), hiddenLayerCounts);

            DeserializeNodes(reader, neuralNetwork);

            return neuralNetwork;
        }

        /// <summary>
        /// Deserialize node-related numeric values (biases and weights) from the reader and assign them to the network.
        /// </summary>
        /// <param name="reader">The <see cref="StreamReader"/> positioned after the header lines.</param>
        /// <param name="neuralNetwork">The network to populate with deserialized values.</param>
        private static void DeserializeNodes(StreamReader reader, INeuralNetwork neuralNetwork)
        {
            DeserializeBiasesAndPreiousBiasDeltasBiasDeltas(reader, neuralNetwork);

            DeserializeWeightsAndPreviousWeightDeltas(reader, neuralNetwork);
        }

        /// <summary>
        /// Read and assign biases and previous bias deltas for all neurons in the network.
        /// </summary>
        /// <param name="reader">The reader to read CSV lines from.</param>
        /// <param name="neuralNetwork">The network whose neurons will be updated.</param>
        private static void DeserializeBiasesAndPreiousBiasDeltasBiasDeltas(StreamReader reader, INeuralNetwork neuralNetwork)
        {
            var allNodes = neuralNetwork.GetAllNodes();

            DeserializeBiases(reader, allNodes);
            DeserializePreviousBiasDeltas(reader, allNodes);
        }

        /// <summary>
        /// Read and assign connection weights and previous weight deltas for all output connections in the network.
        /// </summary>
        /// <param name="reader">The reader to read CSV lines from.</param>
        /// <param name="neuralNetwork">The network whose connections will be updated.</param>
        private static void DeserializeWeightsAndPreviousWeightDeltas(StreamReader reader, INeuralNetwork neuralNetwork)
        {
            var allOutputNodes = neuralNetwork.GetAllOutputConnections();

            DeserializeWeights(reader, allOutputNodes);
            DeserializePreviousWeightDeltas(reader, allOutputNodes);
        }

        /// <summary>
        /// Read previous weight delta values from the reader and apply them to the provided connections in order.
        /// </summary>
        /// <param name="reader">The reader to read the CSV line from.</param>
        /// <param name="allOutputNodes">The connections to populate with previous weight deltas.</param>
        private static void DeserializePreviousWeightDeltas(StreamReader reader, IEnumerable<IConnection> allOutputNodes)
        {
            var previousWeightDeltas = ReadArrayOfDoubles(reader, "previousWeightDeltas");
            previousWeightDeltas.Zip(allOutputNodes, (w, c) => c.PreviousWeightDelta = w).ToArray();
        }

        /// <summary>
        /// Read weight values from the reader and apply them to the provided connections in order.
        /// </summary>
        /// <param name="reader">The reader to read the CSV line from.</param>
        /// <param name="allOutputNodes">The connections to populate with weights.</param>
        private static void DeserializeWeights(StreamReader reader, IEnumerable<IConnection> allOutputNodes)
        {
            var weights = ReadArrayOfDoubles(reader, "weights");
            weights.Zip(allOutputNodes, (w, c) => c.Weight = w).ToArray();
        }

        /// <summary>
        /// Read previous bias delta values from the reader and apply them to the provided neurons in order.
        /// </summary>
        /// <param name="reader">The reader to read the CSV line from.</param>
        /// <param name="allNodes">The neurons to populate with previous bias deltas.</param>
        private static void DeserializePreviousBiasDeltas(StreamReader reader, IEnumerable<INeuron> allNodes)
        {
            var previousBiasDeltas = ReadArrayOfDoubles(reader, "previousBiasDeltas");
            previousBiasDeltas.Zip(allNodes, (b, o) => o.PreviousBiasDelta = b).ToArray();
        }

        /// <summary>
        /// Read bias values from the reader and apply them to the provided neurons in order.
        /// </summary>
        /// <param name="reader">The reader to read the CSV line from.</param>
        /// <param name="allNodes">The neurons to populate with biases.</param>
        private static void DeserializeBiases(StreamReader reader, IEnumerable<INeuron> allNodes)
        {
            var biases = ReadArrayOfDoubles(reader, "biases");
            biases.Zip(allNodes, (b, o) => o.Bias = b).ToArray();
        }

        /// <summary>
        /// Parse the provided string as an integer, throwing <see cref="InvalidDataException"/> if parsing fails.
        /// </summary>
        /// <param name="value">The string to parse.</param>
        /// <param name="paramName">A parameter name used in the exception message if parsing fails.</param>
        /// <returns>The parsed integer value.</returns>
        private static int ReadAsInt(string value, string paramName)
        {
            if (!int.TryParse(value, out var result))
                throw new InvalidDataException($"Invalid integer value for {paramName}.");
            return result;
        }

        /// <summary>
        /// Parse the provided string as a double, throwing <see cref="InvalidDataException"/> if parsing fails.
        /// </summary>
        /// <param name="value">The string to parse.</param>
        /// <param name="paramName">A parameter name used in the exception message if parsing fails.</param>
        /// <returns>The parsed double value.</returns>
        private static double ReadAsDouble(string value, string paramName)
        {
            if (!double.TryParse(value, out var result))
                throw new InvalidDataException($"Invalid double value for {paramName}.");
            return result;
        }

        /// <summary>
        /// Read a single CSV line and parse it into an array of doubles. Returns an empty array if the line is empty.
        /// </summary>
        /// <param name="reader">The reader to read from.</param>
        /// <param name="paramName">A parameter name used to indicate the type of values being read when throwing on parse errors.</param>
        /// <returns>An array of parsed doubles.</returns>
        private static double[] ReadArrayOfDoubles(StreamReader reader, string paramName)
        {
            var line = reader.ReadLine();

            if (string.IsNullOrWhiteSpace(line))
                return [];

            return line.Split(_delimiter).Select(x => ReadAsDouble(x, paramName)).ToArray();
        }

        /// <summary>
        /// Read the hidden layer counts line and parse it into an integer array. Returns an empty array if the line is empty.
        /// </summary>
        /// <param name="reader">The reader to read from.</param>
        /// <returns>An array of integers representing hidden layer node counts.</returns>
        private static int[] ReadHiddenLayerCounts(StreamReader reader)
        {
            var line = reader.ReadLine();

            if (string.IsNullOrWhiteSpace(line))
                return [];  

            return line.Split(_delimiter).Select(x => int.Parse(x)).ToArray();
        }

        /// <summary>
        /// Read and validate the node counts header line (input and output counts).
        /// Throws <see cref="InvalidDataException"/> if the header is missing or invalid.
        /// </summary>
        /// <param name="reader">The reader to read from.</param>
        /// <returns>An array of two strings representing inputCount and outputCount.</returns>
        private static string[] ReadNodeCountLine(StreamReader reader)
        {
            var line1 = reader.ReadLine()?.Split(_delimiter)?.ToArray();

            if (line1 == null || line1.Length < 2)
                throw new InvalidDataException(_invalidHeaderMessage);
            return line1;
        }
    }
}
