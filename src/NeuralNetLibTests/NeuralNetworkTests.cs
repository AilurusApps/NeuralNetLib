using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
    public class NeuralNetworkTests
    {
        [Test]
        public void Constructor_CreatesCorrectLayerSizes()
        {
            var nn = NeuralNetworkFactory.Build(inputCount: 3, outputCount: 2, hiddenLayerCounts: [4, 5]);

            Assert.Multiple(() =>
            {
                Assert.That(nn.Inputs, Has.Length.EqualTo(3));
                Assert.That(nn.Outputs, Has.Length.EqualTo(2));
                Assert.That(nn.HiddenLayers, Has.Length.EqualTo(2));
                Assert.That(nn.HiddenLayers[0], Has.Length.EqualTo(4));
                Assert.That(nn.HiddenLayers[1], Has.Length.EqualTo(5));
            });
        }

        [Test]
        public void Fire_ComputesOutput_WithGivenWeights_NoHiddenLayer()
        {
            // simple network with 2 inputs, 1 output, no hidden layers
            var nn = NeuralNetworkFactory.Build(inputCount: 2, outputCount: 1, hiddenLayerCounts: []);

            Assert.Multiple(() =>
            {
                Assert.That(nn.Inputs?.Length, Is.EqualTo(2));
                Assert.That(nn.Outputs?.Length, Is.EqualTo(1));
            });

            // Set deterministic weights and bias
            nn.Inputs![0].Outputs![0].Weight = 0.5;
            nn.Inputs![1].Outputs![0].Weight = 0.25;
            nn.Outputs[0].Bias = -0.1;

            nn.Fire(1.0, 2.0);

            // Expected using Sigmoid activation on output node
            var expected = SigmoidFunction.Instance.Invoke(0.5 * 1.0 + 0.25 * 2.0 + (-0.1));

            Assert.That(nn.Outputs[0].Value, Is.EqualTo(expected).Within(1e-10));
        }
    }
}