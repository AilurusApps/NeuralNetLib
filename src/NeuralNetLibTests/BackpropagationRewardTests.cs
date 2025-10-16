using AilurusApps.NeuralNetLib;
using Moq;

namespace AilurusApps.NeuralNetLibTests
{
    [TestFixture]
    public class BackpropagationRewardTests
    {
        private const double LearningRate = 0.1;
        private const double Momentum = 0.0; // Simplify with 0 momentum
        private const double ActivationDerivative = 0.5; // f'(net)
        private const double ActualOutput = 0.6;
        private const double TargetOutput = 1.0;
        private const double InputValue = 1.0; // Simplify input for weight update check

        private Backpropagation trainer;
        private Mock<INeuralNetwork> mockNetwork;
        private Mock<INeuron> mockOutputNeuron;
        private Mock<IConnection> mockConnection;
        private Mock<INeuron> mockInputNeuron;

        [SetUp]
        public void Setup()
        {
            // Initialize the Backpropagation trainer
            trainer = new Backpropagation
            {
                LearningRate = LearningRate,
                Momentum = Momentum
            };

            // 1. Set up the Mocks
            mockOutputNeuron = new Mock<INeuron>();
            mockInputNeuron = new Mock<INeuron>();
            mockConnection = new Mock<IConnection>();
            mockNetwork = new Mock<INeuralNetwork>();

            // 2. Configure Output Neuron behavior
            mockOutputNeuron.Setup(n => n.Value).Returns(ActualOutput);
            mockOutputNeuron.Setup(n => n.GetDerivativeValue()).Returns(ActivationDerivative);
            // Setup setter for Gradient property (needed by the Backprop logic)
            mockOutputNeuron.SetupSet(n => n.Gradient = It.IsAny<double>());

            // 3. Configure Connection behavior
            mockConnection.Setup(c => c.InputNode).Returns(mockInputNeuron.Object);
            mockConnection.Setup(c => c.OutputNode).Returns(mockOutputNeuron.Object);
            mockConnection.Setup(c => c.PreviousWeightDelta).Returns(0.0); // Momentum = 0

            // 4. Configure Input/Output layer structure
            mockInputNeuron.Setup(n => n.Value).Returns(InputValue);
            mockInputNeuron.Setup(n => n.Outputs).Returns(new IConnection[] { mockConnection.Object });

            mockNetwork.Setup(n => n.Outputs).Returns(new INeuron[] { mockOutputNeuron.Object });
            mockNetwork.Setup(n => n.Inputs).Returns(new INeuron[] { mockInputNeuron.Object });
            mockNetwork.Setup(n => n.HiddenLayers).Returns(new INeuron[0][]); // No hidden layers for simplicity
        }

        // --------------------------------------------------------------------------------

        [TestCase(1.0, 0.2)] // Standard Backprop: (1.0 - 0.6) * 0.5 * 1.0 = 0.2
        [TestCase(5.0, 1.0)] // Positive Reward: (1.0 - 0.6) * 0.5 * 5.0 = 1.0
        [TestCase(-2.0, -0.4)] // Negative Reward: (1.0 - 0.6) * 0.5 * -2.0 = -0.4
        [TestCase(0.0, 0.0)] // Zero Reward: (1.0 - 0.6) * 0.5 * 0.0 = 0.0
        public void BackpropagateWithReward_CalculatesScaledGradientCorrectly(double reward, double expectedGradient)
        {
            // ACT: Call the BackpropagateWithReward method (assuming it's implemented as suggested)
            trainer.Backpropagate(
                network: mockNetwork.Object,
                reward: reward,
                expectedOutputValues: new double[] { TargetOutput }
            );

            // ASSERT: Verify that the Gradient setter on the output neuron was called with the correct scaled value
            mockOutputNeuron.VerifySet(
                n => n.Gradient = It.Is<double>(g => g == expectedGradient),
                Times.Once(),
                $"The gradient should be scaled by the reward, expected {expectedGradient}, but got a different value."
            );
        }

        // --------------------------------------------------------------------------------

        [TestCase(1.0, 0.02)] // Standard Backprop: Delta = LR * Grad * Input = 0.1 * 0.2 * 1.0 = 0.02
        [TestCase(5.0, 0.1)] // Positive Reward: Delta = 0.1 * 1.0 * 1.0 = 0.1
        [TestCase(-2.0, -0.04)] // Negative Reward: Delta = 0.1 * -0.4 * 1.0 = -0.04
        public void BackpropagateWithReward_UpdatesWeightBasedOnScaledGradient(double reward, double expectedWeightDelta)
        {
            // ARRANGE
            const double initialWeight = 0.5;
            double finalWeight = initialWeight + expectedWeightDelta;

            // Use a local variable to store the weight and ensure the mock is read/write
            double currentWeight = initialWeight;

            // Setup the mock to store the weight when the setter is called
            mockConnection.SetupSet(c => c.Weight = It.IsAny<double>())
                          .Callback<double>(w => currentWeight = w); // <-- Captures the value passed to the setter

            // Setup the mock to return the *current* stored value when the getter is called
            mockConnection.SetupGet(c => c.Weight).Returns(() => currentWeight);

            // Setup other required properties/methods if not already in [SetUp]
            mockConnection.Setup(c => c.PreviousWeightDelta).Returns(0.0);
            // Note: The output neuron gradient is set in the previous Backprop step.
            // We must manually set it here for the weight update test to run correctly.
            // Gradient for this case: (1.0 - 0.6) * 0.5 * reward
            double calculatedGradient = ActivationDerivative * (TargetOutput - ActualOutput) * reward;
            mockOutputNeuron.SetupGet(n => n.Gradient).Returns(calculatedGradient);


            // ACT
            // We assume you are using the TrainWithReward or BackpropagateWithReward methods
            trainer.Backpropagate(
                network: mockNetwork.Object,
                reward: reward,
                expectedOutputValues: new double[] { TargetOutput }
            );

            // ASSERT
            // 1. Verify the final weight value was set
            mockConnection.VerifySet(
                c => c.Weight = It.Is<double>(w =>
                    System.Math.Abs(w - finalWeight) < 1e-9), // Use tolerance for double comparison
                Times.Once(),
                $"The weight update should result in a final weight of {finalWeight}."
            );

            // 2. Verify the PreviousWeightDelta was also stored
            mockConnection.VerifySet(
                c => c.PreviousWeightDelta = It.Is<double>(d =>
                    System.Math.Abs(d - expectedWeightDelta) < 1e-9), // Use tolerance
                Times.Once(),
                $"The PreviousWeightDelta was not set correctly, expected {expectedWeightDelta}."
            );

            // Verify that the internal state changed if SetupGet/Callback was used
            Assert.That(currentWeight, Is.EqualTo(finalWeight).Within(1e-9),
                "Internal state tracking failed or weight update was incorrect.");
        }
    }
}
