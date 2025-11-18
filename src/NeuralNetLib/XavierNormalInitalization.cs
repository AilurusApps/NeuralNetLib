namespace AilurusApps.NeuralNetLib
{
    public class XavierNormalInitalization(Random random) : IWeightInitializationStrategy
    {
        public static readonly XavierNormalInitalization Instance = new(Random.Shared);
        public double GetInitialWeight(int inputCount, int outputCount)
        {
            double stdDev = Math.Sqrt(2.0 / (inputCount + outputCount));
            double[,] weights = new double[inputCount, outputCount];

            double u1 = 1.0 - random.NextDouble(); // uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
    }
}
