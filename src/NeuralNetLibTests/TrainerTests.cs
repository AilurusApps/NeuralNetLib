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
            double[][] inputs =
            [
                [0.0, 0.0], // Expected: 0
                [0.0, 1.0], // Expected: 1
                [1.0, 0.0], // Expected: 1
                [1.0, 1.0]  // Expected: 0
            ];

            double[][] targets =
            [
                [0.0],
                [1.0],
                [1.0],
                [0.0]
            ];

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

        [Test]
        public void Train_AdditionTo10()
        {
            // 1. Define the network structure: 2 inputs, 3 hidden neurons, 1 output
            var network = NeuralNetworkFactory.Build(2, 1, [3]);

            // 2. Define the training data
            var numbers = Enumerable.Range(1, 10).ToArray();

            var allData = (from x in numbers
                           from y in numbers
                           select (x, y))
                       .ToArray();

            var maxInput = (double)numbers.Max();
            var maxTotal = (double)allData.Max(d => d.x + d.y);

            // Split training and validation data, take 10% for validation
            var validationData = allData.OrderBy(numbers => Random.Shared.Next()).Take(allData.Length/10).ToArray();

            var trainingData = allData.Except(validationData).ToDictionary(t => (t.x, t.y), t => new TrainingData([t.x/ maxInput, t.y/ maxInput], [(t.x + t.y)/maxTotal]));

            // 3. Train the network (using parameters like learning rate, momentum)
            var algorithm = new Backpropagation()
            {
                LearningRate = 1.1,
                Momentum = 0.9
            };

            var trainer = new Trainer<(int, int)>(trainingData, algorithm);

            trainer.Retrain(network, tolerance: 0.002, maxIterations: 1000000);

            // 4. Test the network against the training data
            foreach (var data in trainingData)
            {
                network.Fire(data.Value.Inputs);
                var output = Math.Round(network.Outputs[0].Value * maxTotal);
                Assert.That(output, Is.EqualTo(data.Key.x + data.Key.y));
            }

            foreach (var data in validationData)
            {
                network.Fire([data.x/ maxInput, data.y/ maxInput]);
                var output = Math.Round(network.Outputs[0].Value * maxTotal);
                Assert.That(output, Is.EqualTo(data.x + data.y));
            }
        }
    }
}
