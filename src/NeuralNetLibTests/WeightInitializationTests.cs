using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
    [TestFixture]
    public class WeightInitializationTests
    {
        [Test]
        public void XavierNormalInitialization_ShouldProduceCorrectDistribution()
        {
            // Arrange
            var random = new Random(42);
            var xavier = new XavierNormalInitalization(random);
            int inputCount = 9;
            int outputCount = 36;
            int sampleSize = 10000;
            
            // Expected standard deviation for Xavier/Glorot initialization
            double expectedStdDev = Math.Sqrt(2.0 / (inputCount + outputCount));
            
            // Act - Generate many samples
            var weights = new double[sampleSize];
            for (int i = 0; i < sampleSize; i++)
            {
                weights[i] = xavier.GetInitialWeight(inputCount, outputCount);
            }
            
            // Assert - Check mean and standard deviation
            double mean = weights.Average();
            double variance = weights.Select(w => Math.Pow(w - mean, 2)).Average();
            double actualStdDev = Math.Sqrt(variance);
            
            Assert.Multiple(() =>
            {
                Assert.That(mean, Is.EqualTo(0).Within(0.05), 
                    $"Mean should be close to 0, but was {mean:F4}");
                Assert.That(actualStdDev, Is.EqualTo(expectedStdDev).Within(0.02), 
                    $"Standard deviation should be approximately {expectedStdDev:F3}, but was {actualStdDev:F3}");
            });
        }

        [Test]
        public void XavierNormalInitialization_ShouldScaleByLayerSize()
        {
            // Arrange
            var random = new Random(123);
            var xavier = new XavierNormalInitalization(random);
            int sampleSize = 5000;
            
            // Test different layer sizes
            var layerConfigs = new[]
            {
                (inputCount: 9, outputCount: 36),   // Small input to larger hidden
                (inputCount: 36, outputCount: 36),  // Same size layers
                (inputCount: 36, outputCount: 9)    // Larger hidden to small output
            };
            
            foreach (var (inputCount, outputCount) in layerConfigs)
            {
                // Act
                var weights = new double[sampleSize];
                for (int i = 0; i < sampleSize; i++)
                {
                    weights[i] = xavier.GetInitialWeight(inputCount, outputCount);
                }
                
                // Assert
                double expectedStdDev = Math.Sqrt(2.0 / (inputCount + outputCount));
                double variance = weights.Select(w => Math.Pow(w - weights.Average(), 2)).Average();
                double actualStdDev = Math.Sqrt(variance);
                
                Assert.That(actualStdDev, Is.EqualTo(expectedStdDev).Within(0.03),
                    $"For [{inputCount}?{outputCount}], stdDev should be {expectedStdDev:F3}, but was {actualStdDev:F3}");
            }
        }

        [Test]
        public void XavierNormalInitialization_ShouldProduceReasonableRange()
        {
            // Arrange
            var random = new Random(789);
            var xavier = new XavierNormalInitalization(random);
            int inputCount = 9;
            int outputCount = 36;
            int sampleSize = 10000;
            
            double expectedStdDev = Math.Sqrt(2.0 / (inputCount + outputCount));
            double reasonableMax = expectedStdDev * 4; // ~99.99% of normal distribution within 4 std devs
            
            // Act
            var weights = new double[sampleSize];
            for (int i = 0; i < sampleSize; i++)
            {
                weights[i] = xavier.GetInitialWeight(inputCount, outputCount);
            }
            
            // Assert - Most weights should be within reasonable range
            var outliers = weights.Count(w => Math.Abs(w) > reasonableMax);
            double outlierPercentage = (double)outliers / sampleSize * 100;
            
            Assert.That(outlierPercentage, Is.LessThan(1.0),
                $"Less than 1% of weights should be outliers, but {outlierPercentage:F2}% were beyond ±{reasonableMax:F2}");
        }

        [Test]
        public void XavierNormalInitialization_ShouldPreventSaturation()
        {
            // Arrange - Test that initial weights won't cause sigmoid saturation
            var random = new Random(456);
            var xavier = new XavierNormalInitalization(random);
            int inputCount = 9;
            int outputCount = 9;
            int sampleSize = 1000;
            
            // For sigmoid activation, we want to avoid values that cause saturation (|x| > 4)
            // With proper initialization, the weighted sum should typically be in a safe range
            
            // Act - Simulate a layer's weighted sum with maximum input values (all 1s or all -1s)
            var maxWeightedSums = new double[sampleSize];
            for (int i = 0; i < sampleSize; i++)
            {
                var weights = new double[inputCount];
                for (int j = 0; j < inputCount; j++)
                {
                    weights[j] = xavier.GetInitialWeight(inputCount, outputCount);
                }
                // Worst case: all inputs are 1 (or -1)
                maxWeightedSums[i] = Math.Abs(weights.Sum());
            }
            
            // Assert - Most weighted sums should be reasonable (won't saturate sigmoid)
            var saturatingCases = maxWeightedSums.Count(sum => sum > 4.0);
            double saturatingPercentage = (double)saturatingCases / sampleSize * 100;
            
            Assert.That(saturatingPercentage, Is.LessThan(5.0),
                $"Less than 5% should risk saturation, but {saturatingPercentage:F1}% had weighted sums > 4");
        }

        [Test]
        public void NetworkWithXavierInit_ShouldNotSaturateImmediately()
        {
            // Arrange - Create a realistic network like TicTacToe uses
            var network = NeuralNetworkFactory.Build(
                inputCount: 9, 
                outputCount: 9, 
                hiddenLayerCounts: [36, 36],
                activationFunction: HyperTanFunction.Instance,
                outputLayerFunction: SigmoidFunction.Instance,
                weightInitializationStrategy: XavierNormalInitalization.Instance);
            
            // Act - Fire with various input patterns
            var testInputs = new[]
            {
                new double[] { 1, 0, 0, 0, -1, 0, 0, 0, 0 },    // Typical game state
                new double[] { 1, 1, 1, -1, -1, -1, 0, 0, 0 },  // Mid-game
                new double[] { 1, -1, 1, -1, 1, -1, 1, -1, 1 }  // Full board
            };
            
            foreach (var input in testInputs)
            {
                network.Fire(input);
                
                // Assert - Outputs should not be saturated (too close to 0 or 1)
                var outputs = network.Outputs.Select(o => o.Value).ToArray();
                var saturatedOutputs = outputs.Count(o => o < 0.01 || o > 0.99);
                
                Assert.That(saturatedOutputs, Is.LessThan(9),
                    $"Some outputs should be in learnable range (not all saturated) for input [{string.Join(",", input)}]");
                
                // Check that we have reasonable variety in outputs
                var minOutput = outputs.Min();
                var maxOutput = outputs.Max();
                var range = maxOutput - minOutput;
                
                Assert.That(range, Is.GreaterThan(0.01),
                    $"Output range should show some variety, but was only {range:F3} for input [{string.Join(",", input)}]");
            }
        }

        [Test]
        public void XavierNormalInitialization_ShouldBeDeterministicWithSameSeed()
        {
            // Arrange
            const int seed = 12345;
            var xavier1 = new XavierNormalInitalization(new Random(seed));
            var xavier2 = new XavierNormalInitalization(new Random(seed));
            
            int inputCount = 10;
            int outputCount = 20;
            int sampleSize = 100;
            
            // Act
            var weights1 = new double[sampleSize];
            var weights2 = new double[sampleSize];
            
            for (int i = 0; i < sampleSize; i++)
            {
                weights1[i] = xavier1.GetInitialWeight(inputCount, outputCount);
                weights2[i] = xavier2.GetInitialWeight(inputCount, outputCount);
            }
            
            // Assert - Sequences should be identical
            Assert.That(weights2, Is.EqualTo(weights1),
                "Same seed should produce identical weight sequences");
        }

        [Test]
        public void XavierNormalInitialization_ShouldProduceDifferentValuesWithDifferentSeeds()
        {
            // Arrange
            var xavier1 = new XavierNormalInitalization(new Random(111));
            var xavier2 = new XavierNormalInitalization(new Random(222));
            
            int inputCount = 10;
            int outputCount = 20;
            int sampleSize = 100;
            
            // Act
            var weights1 = new double[sampleSize];
            var weights2 = new double[sampleSize];
            
            for (int i = 0; i < sampleSize; i++)
            {
                weights1[i] = xavier1.GetInitialWeight(inputCount, outputCount);
                weights2[i] = xavier2.GetInitialWeight(inputCount, outputCount);
            }
            
            // Assert - Sequences should be different
            var differentCount = weights1.Zip(weights2, (a, b) => a != b).Count(isDifferent => isDifferent);
            
            Assert.That(differentCount, Is.GreaterThan(90),
                $"Different seeds should produce different sequences (at least 90% different), but only {differentCount}/100 were different");
        }

        [Test]
        public void XavierNormalInitialization_CompareWithUnscaledNormal()
        {
            // Arrange - This test documents the bug fix
            var random = new Random(42);
            var xavier = new XavierNormalInitalization(random);
            
            int inputCount = 9;
            int outputCount = 36;
            int sampleSize = 5000;
            
            double expectedStdDev = Math.Sqrt(2.0 / (inputCount + outputCount));
            
            // Act
            var weights = new double[sampleSize];
            for (int i = 0; i < sampleSize; i++)
            {
                weights[i] = xavier.GetInitialWeight(inputCount, outputCount);
            }
            
            double actualStdDev = Math.Sqrt(weights.Select(w => w * w).Average());
            
            // Assert - Should be scaled, not standard normal (stddev ? 1.0)
            Assert.That(actualStdDev, Is.LessThan(0.5),
                $"Xavier weights should be scaled down from standard normal. Expected ~{expectedStdDev:F3}, got {actualStdDev:F3}");
            
            Assert.That(actualStdDev, Is.GreaterThan(0.1),
                $"Xavier weights should not be too small. Expected ~{expectedStdDev:F3}, got {actualStdDev:F3}");
        }

        [Test]
        public void NetworkFactory_ShouldUseXavierInitByDefault()
        {
            // Arrange & Act
            var network = NeuralNetworkFactory.Build(
                inputCount: 9,
                outputCount: 9,
                hiddenLayerCounts: [18]);
            
            // Collect all weights from the network
            var allWeights = new List<double>();
            
            // Input layer weights
            foreach (var input in network.Inputs)
            {
                if (input.Outputs != null)
                {
                    allWeights.AddRange(input.Outputs.Select(c => c.Weight));
                }
            }
            
            // Hidden layer weights
            foreach (var layer in network.HiddenLayers)
            {
                foreach (var neuron in layer)
                {
                    if (neuron.Outputs != null)
                    {
                        allWeights.AddRange(neuron.Outputs.Select(c => c.Weight));
                    }
                }
            }
            
            // Assert - Weights should have Xavier-like distribution
            if (allWeights.Count > 0)
            {
                double mean = allWeights.Average();
                double stdDev = Math.Sqrt(allWeights.Select(w => Math.Pow(w - mean, 2)).Average());
                
                Assert.Multiple(() =>
                {
                    Assert.That(mean, Is.EqualTo(0).Within(0.1),
                        "Mean of initial weights should be near 0");
                    Assert.That(stdDev, Is.LessThan(0.5),
                        "Weights should be scaled (not standard normal with stddev ? 1)");
                    Assert.That(stdDev, Is.GreaterThan(0.1),
                        "Weights should not be too small");
                });
            }
        }
    }
}
