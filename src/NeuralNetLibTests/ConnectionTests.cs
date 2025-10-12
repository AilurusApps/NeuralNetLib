using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
    public class ConnectionTests
    {
        [Test]
        public void Connection_Properties_CanBeSetAndRetrieved()
        {
            var inputNode = new Neuron(HyperTanFunction.Instance, 1);
            var outputNode = new Neuron(SigmoidFunction.Instance, 1);
            double weight = 0.5;
            double previousWeightDelta = 0.1;

            var connection = new Connection(inputNode, weight, outputNode)
            {
                PreviousWeightDelta = previousWeightDelta
            };

            Assert.Multiple(() =>
            {
                Assert.That(connection.InputNode, Is.EqualTo(inputNode));
                Assert.That(connection.OutputNode, Is.EqualTo(outputNode));
                Assert.That(connection.Weight, Is.EqualTo(weight));
                Assert.That(connection.PreviousWeightDelta, Is.EqualTo(previousWeightDelta));
            });
        }
    }
}