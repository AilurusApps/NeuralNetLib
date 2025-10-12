namespace AilurusApps.NeuralNetLib
{
    /// <summary>
    /// Represents a directed, weighted connection between two neurons in the network.
    /// Connections carry a value from <see cref="InputNode"/> to <see cref="OutputNode"/> scaled by <see cref="Weight"/>.
    /// Implementations store the last weight delta in <see cref="PreviousWeightDelta"/> to support momentum-based training algorithms.
    /// </summary>
    public interface IConnection
    {
        /// <summary>
        /// The neuron that provides the input value for this connection.
        /// </summary>
        INeuron InputNode { get; set; }

        /// <summary>
        /// The neuron that receives the weighted input value from this connection.
        /// </summary>
        INeuron OutputNode { get; set; }

        /// <summary>
        /// The last change applied to the weight. Training algorithms use this value when applying momentum.
        /// </summary>
        double PreviousWeightDelta { get; set; }

        /// <summary>
        /// The weight applied to the input value when propagating to the output neuron.
        /// </summary>
        double Weight { get; set; }
    }
}