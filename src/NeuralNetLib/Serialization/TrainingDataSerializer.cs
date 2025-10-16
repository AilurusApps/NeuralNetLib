using System.Text;

namespace AilurusApps.NeuralNetLib.Serialization
{
    /// <summary>
    /// Serializes collections of <see cref="TrainingData"/> instances to text streams or files.
    /// Each training example is written as a single line with inputs and outputs separated by a semicolon
    /// and individual values separated by commas (e.g. "0.1,0.2;1.0").
    /// </summary>
    public class TrainingDataSerializer
    {
        /// <summary>
        /// Serialize a sequence of training examples to the provided stream using a simple text format.
        /// The stream is not closed by this method; a <see cref="StreamWriter"/> is used for the duration of the call.
        /// </summary>
        /// <param name="trainingDataList">The training data to serialize.</param>
        /// <param name="stream">The stream to write serialized data to.</param>
        public static void SerializeToStream(IEnumerable<TrainingData> trainingDataList, Stream stream)
        {
            using var writer = new StreamWriter(stream, Encoding.Default);

            foreach (var trainingData in trainingDataList)
            {
                writer.WriteLine(SerializeToString(trainingData));
            }
        }

        /// <summary>
        /// Serializes a training data record into a string.
        /// </summary>
        /// <param name="trainingData">The training data record to serialize</param>
        /// <returns>A string the data is serialized to.</returns>
        private static string SerializeToString(TrainingData trainingData)
        {
            var line = string.Join(",", trainingData.Inputs) + ";" + string.Join(",", trainingData.Outputs);

            if (trainingData.Reward != null)
                line += $";{trainingData.Reward.Value}";

            return line;
        }

        /// <summary>
        /// Serialize a sequence of training examples to a file at the specified path.
        /// The file will be opened for writing and overwritten if it exists.
        /// </summary>
        /// <param name="trainingDataList">The training data to serialize.</param>
        /// <param name="filePath">The target file path to write serialized content into.</param>
        public static void SerializeToFile(IEnumerable<TrainingData> trainingDataList, string filePath)
        {
            using var file = File.OpenWrite(filePath);
            SerializeToStream(trainingDataList, file);
        }
    }
}
