namespace AilurusApps.NeuralNetLib
{
    public interface IWeightInitializationStrategy
    {
        double GetInitialWeight(int inputCount, int outputCount);
    }
}
