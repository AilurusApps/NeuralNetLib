using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
    public class TrainerTests
    {
        [Test]
        public void Train_XOR()
        {
            // 1. Define the network structure: 2 inputs, 3 hidden neurons, 1 output
            var network = NeuralNetworkFactory.Build(2, 1, [3]);

            // 2. Define the training data (XOR Logic)
            double[][] inputs = new double[][]
            {
            new[] { 0.0, 0.0 }, // Expected: 0
            new[] { 0.0, 1.0 }, // Expected: 1
            new[] { 1.0, 0.0 }, // Expected: 1
            new[] { 1.0, 1.0 }  // Expected: 0
            };

            double[][] targets = new double[][]
            {
            new[] { 0.0 },
            new[] { 1.0 },
            new[] { 1.0 },
            new[] { 0.0 }
            };

            // 3. Train the network (using parameters like learning rate, momentum)
            var algorithm = new Backpropagation()
            {
                LearningRate = 0.2,
                Momentum = 0.1
            };

            var trainingData = inputs.Zip(targets).ToDictionary(t => (t.First[0], t.First[1]), t => new TrainingData(t.First, t.Second));

            var trainer = new Trainer<(double,double)>(trainingData, algorithm);

            trainer.Retrain(network, tolerance: 0, maxIterations: 10000);

            // 4. Test the network
            foreach (var data in trainingData.Values)
            {
                network.Fire(data.Inputs);
                var output = Math.Round(network.Outputs[0].Value);
                Assert.That(output, Is.EqualTo(data.Outputs[0]));
            }
        }
    }
}
