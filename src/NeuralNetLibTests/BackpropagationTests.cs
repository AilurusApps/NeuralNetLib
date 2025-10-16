using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
    public class BackpropagationTests
    {

        [Test]
        public void Train_BackpropagatesAndUpdatesWeightsAndBias_NoHiddenLayer()
        {
            // 2 inputs -> 1 output, no hidden layers
            var nn = NeuralNetworkFactory.Build(inputCount: 2, outputCount: 1, hiddenLayerCounts: []);

            Assert.Multiple(() =>
            {
                Assert.That(nn.Inputs?.Length, Is.EqualTo(2));
                Assert.That(nn.Outputs?.Length, Is.EqualTo(1));
            });

            // deterministic initial values
            var initialW0 = 0.5;
            var initialW1 = -0.5;
            var initialBias = 0.0;

            nn.Inputs![0].Outputs![0].Weight = initialW0;
            nn.Inputs![1].Outputs![0].Weight = initialW1;
            nn.Outputs[0].Bias = initialBias;

            var backpropagation = new Backpropagation()
            {
                LearningRate = 0.1,
                Momentum = 0.0
            };

            var input0 = 1.0;
            var input1 = 0.0;
            var target = 1.0;

            // Compute expected values based on initial parameters (before training)
            var netInput = initialW0 * input0 + initialW1 * input1 + initialBias;
            var outputValue = SigmoidFunction.Instance.Invoke(netInput);
            var derivative = SigmoidFunction.Instance.GetDerivativeValue(outputValue);
            var gradient = derivative * (target - outputValue);

            var expectedDeltaW0 = backpropagation.LearningRate * gradient * input0;
            var expectedNewW0 = initialW0 + expectedDeltaW0;

            var expectedDeltaBias = backpropagation.LearningRate * gradient;
            var expectedNewBias = initialBias + expectedDeltaBias;

            backpropagation.Train(nn, [input0, input1], Constants.DefaultReward, target);

            Assert.Multiple(() =>
            {
                // weights and bias updated as expected (within tolerance)
                Assert.That(nn.Inputs![0].Outputs![0].Weight, Is.EqualTo(expectedNewW0).Within(1e-6));
                Assert.That(nn.Outputs[0].Bias, Is.EqualTo(expectedNewBias).Within(1e-6));
            });
        }

    }
}
