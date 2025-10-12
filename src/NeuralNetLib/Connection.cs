namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents a weighted connection between two neurons in the network.
    /// The connection carries a value from the <see cref="InputNode"/> to the <see cref="OutputNode"/>
    /// multiplied by the <see cref="Weight"/>. PreviousWeightDelta is used to support momentum during training.
    /// </summary>
    /// <param name="inputNode">The source neuron that provides the value for this connection.</param>
    /// <param name="weight">The initial weight for the connection.</param>
    /// <param name="outputNode">The target neuron that receives the weighted input.</param>
    public class Connection(INeuron inputNode, double weight, INeuron outputNode) : IConnection
    {
    /// <summary>
    /// The neuron that provides the input value for this connection.
    /// </summary>
    public INeuron InputNode { get; set; } = inputNode;

    /// <summary>
    /// The weight applied to the input value when propagating to the output neuron.
    /// </summary>
    public double Weight { get; set; } = weight;

    /// <summary>
    /// The previous weight delta used by learning algorithms that use momentum.
    /// </summary>
    public double PreviousWeightDelta { get; set; }

    /// <summary>
    /// The neuron that receives the weighted input from this connection.
    /// </summary>
    public INeuron OutputNode { get; set; } = outputNode;
    }
}
