namespace AilurusApps.NeuralNetLib
{
    public class XavierUniformInitialization(Random random) : IWeightInitializationStrategy
    {
        public double GetInitialWeight(int inputCount, int outputCount)
        {
            double limit = Math.Sqrt(6.0 / (inputCount + outputCount));

            return random.NextDouble() * 2 * limit - limit; // Uniformly distributed between -limit and limit
        }
    }
}
