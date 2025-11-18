namespace AilurusApps.NeuralNetLib
{
    public class RandomWeightInitialization(Random random) : IWeightInitializationStrategy
    {
        public static readonly RandomWeightInitialization Instance = new(Random.Shared);
        public double GetInitialWeight(int inputCount, int outputCount)
        {
            return 0.49 + (random.NextDouble() * 0.02);
        }
    }
}
