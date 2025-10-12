using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
    [TestFixture]
    public class NeuralNetworkFactoryTests
    {
        [Test]
        public void Build_NoHiddenLayers_CreatesCorrectTopology()
        {
            // Arrange
            const int inputCount = 3;
            const int outputCount = 2;
            int[] hiddenLayerCounts = Array.Empty<int>();

            // Act
            var network = (NeuralNetwork)NeuralNetworkFactory.Build(inputCount, outputCount, hiddenLayerCounts);

            Assert.Multiple(() =>
            {
                // Assert - Layer Counts
                Assert.That(network.Inputs.Length, Is.EqualTo(inputCount), "Input layer count is incorrect.");
                Assert.That(network.Outputs.Length, Is.EqualTo(outputCount), "Output layer count is incorrect.");
                Assert.That(network.HiddenLayers.Length, Is.EqualTo(0), "Hidden layers array should be empty.");

                // Assert - Activation Functions
                Assert.That(network.Inputs.All(n => n.ActivationFunction is HyperTanFunction), "Input neurons should use HyperTanFunction.");
            });
            Assert.That(network.Outputs.All(n => n.ActivationFunction is SigmoidFunction), "Output neurons should use SigmoidFunction.");

            // Assert - Connections (Input to Output)
            var inputNeuron = network.Inputs.First();
            Assert.That(inputNeuron.Outputs!.Length, Is.EqualTo(outputCount), "Input neuron should have connections to all output neurons.");

            var outputNeuron = network.Outputs.First();
            Assert.That(outputNeuron.Inputs!.Length, Is.EqualTo(inputCount), "Output neuron should have connections from all input neurons.");
        }

        [Test]
        public void Build_OneHiddenLayer_CreatesCorrectTopologyAndWiring()
        {
            // Arrange
            const int inputCount = 2;
            const int hiddenCount = 4;
            const int outputCount = 1;
            int[] hiddenLayerCounts = new[] { hiddenCount };

            // Act
            var network = (NeuralNetwork)NeuralNetworkFactory.Build(inputCount, outputCount, hiddenLayerCounts);

            Assert.Multiple(() =>
            {
                // Assert - Layer Counts
                Assert.That(network.Inputs.Length, Is.EqualTo(inputCount), "Input layer count is incorrect.");
                Assert.That(network.Outputs.Length, Is.EqualTo(outputCount), "Output layer count is incorrect.");
                Assert.That(network.HiddenLayers.Length, Is.EqualTo(1), "Should have exactly one hidden layer.");
                Assert.That(network.HiddenLayers[0].Length, Is.EqualTo(hiddenCount), "Hidden layer count is incorrect.");
            });
            Assert.Multiple(() =>
            {
                // Assert - Activation Functions
                Assert.That(network.Inputs.All(n => n.ActivationFunction is HyperTanFunction), "Input neurons should use HyperTanFunction.");
                Assert.That(network.HiddenLayers[0].All(n => n.ActivationFunction is HyperTanFunction), "Hidden neurons should use HyperTanFunction.");
                Assert.That(network.Outputs.All(n => n.ActivationFunction is SigmoidFunction), "Output neurons should use SigmoidFunction.");
            });

            // Assert - Connections (Input to Hidden)
            var inputToHiddenConnection = network.Inputs.First().Outputs!.First();
            Assert.Multiple(() =>
            {
                Assert.That(inputToHiddenConnection.InputNode, Is.EqualTo(network.Inputs.First()), "Input to Hidden connection source is wrong.");
                Assert.That(inputToHiddenConnection.OutputNode, Is.EqualTo(network.HiddenLayers[0].First()), "Input to Hidden connection target is wrong.");
            });

            // Assert - Connections (Hidden to Output)
            var hiddenToOutputConnection = network.HiddenLayers[0].First().Outputs!.First();
            Assert.Multiple(() =>
            {
                Assert.That(hiddenToOutputConnection.InputNode, Is.EqualTo(network.HiddenLayers[0].First()), "Hidden to Output connection source is wrong.");
                Assert.That(hiddenToOutputConnection.OutputNode, Is.EqualTo(network.Outputs.First()), "Hidden to Output connection target is wrong.");
            });
        }

        [Test]
        public void Build_MultipleHiddenLayers_CorrectConnectionFlow()
        {
            // Arrange
            const int iCount = 1;
            const int h1Count = 2;
            const int h2Count = 3;
            const int oCount = 4;
            int[] hiddenLayerCounts = new[] { h1Count, h2Count };

            // Act
            var network = (NeuralNetwork)NeuralNetworkFactory.Build(iCount, oCount, hiddenLayerCounts);

            // Assert - Layer Counts
            Assert.That(network.HiddenLayers.Length, Is.EqualTo(2), "Should have exactly two hidden layers.");
            Assert.That(network.HiddenLayers[0].Length, Is.EqualTo(h1Count), "First hidden layer count is incorrect.");
            Assert.That(network.HiddenLayers[1].Length, Is.EqualTo(h2Count), "Second hidden layer count is incorrect.");

            // Assert - Connection Layer Sizes
            var inputLayer = network.Inputs;
            var hiddenLayer1 = network.HiddenLayers[0];
            var hiddenLayer2 = network.HiddenLayers[1];
            var outputLayer = network.Outputs;

            // I -> H1 connection size check (1 input connects to 2 hidden)
            Assert.That(inputLayer.First().Outputs!.Length, Is.EqualTo(h1Count), "I to H1 output size is wrong.");
            Assert.That(hiddenLayer1.First().Inputs!.Length, Is.EqualTo(iCount), "I to H1 input size is wrong.");

            // H1 -> H2 connection size check (2 hidden connect to 3 hidden)
            Assert.That(hiddenLayer1.First().Outputs!.Length, Is.EqualTo(h2Count), "H1 to H2 output size is wrong.");
            Assert.That(hiddenLayer2.First().Inputs!.Length, Is.EqualTo(h1Count), "H1 to H2 input size is wrong.");

            // H2 -> O connection size check (3 hidden connect to 4 output)
            Assert.That(hiddenLayer2.First().Outputs!.Length, Is.EqualTo(oCount), "H2 to O output size is wrong.");
            Assert.That(outputLayer.First().Inputs!.Length, Is.EqualTo(h2Count), "H2 to O input size is wrong.");
        }

        [Test]
        public void Build_InterLayerWiring_InputsAndOutputsAreReciprocal()
        {
            // Arrange
            const int inputCount = 2;
            const int outputCount = 3;
            int[] hiddenLayerCounts = new[] { 2 };

            // Act
            var network = (NeuralNetwork)NeuralNetworkFactory.Build(inputCount, outputCount, hiddenLayerCounts);

            var inputNeuron1 = network.Inputs[0];
            var hiddenNeuron1 = network.HiddenLayers[0][0];

            // Assert - Check connectivity from Input 1 to Hidden 1
            var inputToHiddenConnection = inputNeuron1.Outputs!.First(c => c.OutputNode == hiddenNeuron1);

            // Assert - Reciprocal check: Does Hidden 1 have this connection in its Inputs array?
            // The connection should be at index 0 of the hidden neuron's inputs array, as Input 1 is index 0.
            Assert.That(hiddenNeuron1.Inputs![0], Is.EqualTo(inputToHiddenConnection),
                "Hidden neuron input array should contain the connection from the source neuron at the correct index.");
        }
    }

}
