using AilurusApps.NeuralNetLib.Extensions;
using System.Text;

namespace AilurusApps.NeuralNetLib.Serialization
{
    /// <summary>
    /// Reads training examples from a text file or stream and converts them into <see cref="TrainingData"/> instances.
    /// Each non-empty line in the input is expected to contain input and output vectors separated by a semicolon
    /// (e.g. "0.1,0.2;1.0").
    /// </summary>
    public class TrainingDataDeserializer
    {
        /// <summary>
        /// Deserialize training data from a file. Returns an empty list if the file does not exist.
        /// </summary>
        /// <param name="filePath">Path to the file containing training examples.</param>
        /// <returns>A list of <see cref="TrainingData"/> read from the file.</returns>
        public static IList<TrainingData> DeserializeFromFile(string filePath)
        {
            if (!File.Exists(filePath))
                return [];

            using var stream = File.OpenRead(filePath);

            return DeserializeFromStream(stream).ToArray();
        }

        /// <summary>
        /// Deserialize training data from a stream. Lines that are null or whitespace are ignored.
        /// Each line must contain two ';' separated sections: inputs and outputs.
        /// Inputs and outputs are comma-separated numeric values.
        /// </summary>
        /// <param name="stream">The stream to read training data from.</param>
        /// <returns>An enumerable of <see cref="TrainingData"/> instances.</returns>
        public static IEnumerable<TrainingData> DeserializeFromStream(Stream stream)
        {
            using var reader = new StreamReader(stream, Encoding.Default);

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var split = line.Split(';');

                if (split.Length < 2)
                    continue;

                yield return new TrainingData(GetInputs(split), GetOutputs(split))
                {
                    Reward = GetReward(split)
                };
            }
        }

        /// <summary>
        /// Parse the outputs portion of a split line into a double array.
        /// Expects the second segment (index 1) to contain comma-separated values.
        /// </summary>
        /// <param name="split">The line split by ';'.</param>
        /// <returns>An array of output values.</returns>
        private static double[] GetOutputs(string[] split)
        {
            return split[1].Split(',').Select(s => s.ReadAsDouble("output")).ToArray();
        }

        /// <summary>
        /// Parse the inputs portion of a split line into a double array.
        /// Expects the first segment (index 0) to contain comma-separated values.
        /// </summary>
        /// <param name="split">The line split by ';'.</param>
        /// <returns>An array of input values.</returns>
        private static double[] GetInputs(string[] split)
        {
            return split[0].Split(',').Select(s => s.ReadAsDouble("input")).ToArray();
        }

        /// <summary>
        /// Parse the reward portion of a split line into a double value.
        /// </summary>
        /// <param name="split">The line split by ';'.</param>
        /// <returns>The double representing the reward value. null a reward has not been specified.</returns>
        private static double? GetReward(string[] split)
        {
            if (split.Length < 3)
                return null;

            return split[2].ReadAsDouble("reward");
        }
    }
}
