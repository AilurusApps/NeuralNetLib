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

        [Test]
        public void TrainUntil_ShouldStopWhenCompletionFuncReturnsTrue()
        {
            // Arrange
            var network = NeuralNetworkFactory.Build(2, 1, [3]);
            var algorithm = new Backpropagation()
            {
                LearningRate = 0.2,
                Momentum = 0.1
            };

            var trainingData = new Dictionary<string, TrainingData>();
            var trainer = new Trainer<string>(trainingData, algorithm);

            var data = new TrainingData([0.5, 0.5], [1.0]);
            
            int iterationCount = 0;
            const int stopAfterIterations = 5;
            
            // Act
            var result = trainer.TrainUntil(network, maxIterations: 100, data, () =>
            {
                iterationCount++;
                return iterationCount >= stopAfterIterations;
            });

            // Assert
            Assert.That(result, Is.True, "TrainUntil should return true when completion func returns true");
            Assert.That(iterationCount, Is.EqualTo(stopAfterIterations), "Should have executed exactly the expected number of iterations");
        }

        [Test]
        public void TrainUntil_ShouldReturnFalseWhenMaxIterationsReached()
        {
            // Arrange
            var network = NeuralNetworkFactory.Build(2, 1, [3]);
            var algorithm = new Backpropagation()
            {
                LearningRate = 0.2,
                Momentum = 0.1
            };

            var trainingData = new Dictionary<string, TrainingData>();
            var trainer = new Trainer<string>(trainingData, algorithm);

            var data = new TrainingData([0.5, 0.5], [1.0]);
            
            int iterationCount = 0;
            const int maxIterations = 10;
            
            // Act - completion func never returns true
            var result = trainer.TrainUntil(network, maxIterations, data, () =>
            {
                iterationCount++;
                return false; // Never complete
            });

            // Assert
            Assert.That(result, Is.False, "TrainUntil should return false when maxIterations reached without completion");
            Assert.That(iterationCount, Is.EqualTo(maxIterations), "Should have executed exactly maxIterations");
        }

        [Test]
        public void TrainUntil_ShouldStopImmediatelyWhenCompletionFuncStartsTrue()
        {
            // Arrange
            var network = NeuralNetworkFactory.Build(2, 1, [3]);
            var algorithm = new Backpropagation()
            {
                LearningRate = 0.2,
                Momentum = 0.1
            };

            var trainingData = new Dictionary<string, TrainingData>();
            var trainer = new Trainer<string>(trainingData, algorithm);

            var data = new TrainingData([0.5, 0.5], [1.0]);
            
            int iterationCount = 0;
            
            // Act - completion func returns true immediately
            var result = trainer.TrainUntil(network, maxIterations: 100, data, () =>
            {
                iterationCount++;
                return true; // Always complete
            });

            // Assert
            Assert.That(result, Is.True, "TrainUntil should return true when completion func returns true");
            Assert.That(iterationCount, Is.EqualTo(1), "Should have executed only 1 iteration");
        }

        [Test]
        public void TrainUntil_ShouldUpdateNetworkWeights()
        {
            // Arrange
            var network = NeuralNetworkFactory.Build(2, 1, [2]);
            var algorithm = new Backpropagation()
            {
                LearningRate = 0.5,
                Momentum = 0.1
            };

            var trainingData = new Dictionary<string, TrainingData>();
            var trainer = new Trainer<string>(trainingData, algorithm);

            var data = new TrainingData([1.0, 0.0], [1.0]);
            
            // Get initial output
            network.Fire(data.Inputs);
            var initialOutput = network.Outputs[0].Value;
            
            // Act - train for a few iterations
            trainer.TrainUntil(network, maxIterations: 10, data, () => false);

            // Assert - output should have changed
            network.Fire(data.Inputs);
            var finalOutput = network.Outputs[0].Value;
            
            Assert.That(finalOutput, Is.Not.EqualTo(initialOutput), 
                "Network output should change after training");
        }

        [Test]
        public void TrainUntil_ShouldUseRewardFromTrainingData()
        {
            // Arrange
            var network = NeuralNetworkFactory.Build(2, 1, [3]);
            var algorithm = new Backpropagation()
            {
                LearningRate = 0.5,
                Momentum = 0.1
            };

            var trainingData = new Dictionary<string, TrainingData>();
            var trainer = new Trainer<string>(trainingData, algorithm);

            // Create data with custom reward
            var data = new TrainingData([0.5, 0.5], [1.0])
            {
                Reward = 2.0
            };
            
            // Act - should not throw and should use the reward
            Assert.DoesNotThrow(() =>
            {
                trainer.TrainUntil(network, maxIterations: 5, data, () => false);
            });
        }

        [Test]
        public void TrainUntil_ShouldHandleErrorBasedCompletion()
        {
            // Arrange
            var network = NeuralNetworkFactory.Build(2, 1, [3]);
            var algorithm = new Backpropagation()
            {
                LearningRate = 0.3,
                Momentum = 0.1
            };

            var trainingData = new Dictionary<string, TrainingData>();
            var trainer = new Trainer<string>(trainingData, algorithm);

            var data = new TrainingData([0.0, 1.0], [1.0]);
            
            const double targetError = 0.1;
            
            // Act - stop when error is below threshold
            var result = trainer.TrainUntil(network, maxIterations: 1000, data, () =>
            {
                network.Fire(data.Inputs);
                var error = Math.Abs(network.Outputs[0].Value - data.Outputs[0]);
                return error < targetError;
            });

            // Assert
            Assert.That(result, Is.True, "Should achieve target error");
            
            network.Fire(data.Inputs);
            var finalError = Math.Abs(network.Outputs[0].Value - data.Outputs[0]);
            Assert.That(finalError, Is.LessThan(targetError), "Final error should be below threshold");
        }

        [Test]
        public void TrainUntil_ShouldRespectMaxIterationsWithDifferentCompletionLogic()
        {
            // Arrange
            var network = NeuralNetworkFactory.Build(2, 1, [3]);
            var algorithm = new Backpropagation()
            {
                LearningRate = 0.2,
                Momentum = 0.1
            };

            var trainingData = new Dictionary<string, TrainingData>();
            var trainer = new Trainer<string>(trainingData, algorithm);

            var data = new TrainingData([0.5, 0.5], [0.8]);
            
            int callCount = 0;
            const int maxIterations = 15;
            
            // Act - completion func alternates but never completes
            var result = trainer.TrainUntil(network, maxIterations, data, () =>
            {
                callCount++;
                return callCount > 100; // Impossible to reach within maxIterations
            });

            // Assert
            Assert.That(result, Is.False, "Should return false when maxIterations reached");
            Assert.That(callCount, Is.EqualTo(maxIterations), "Completion func should be called exactly maxIterations times");
        }
    }
}
