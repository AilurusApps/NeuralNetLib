namespace AilurusApps.NeuralNetLib.Extensions
{
    /// <summary>
    /// Extension methods for <see cref="string"/>.
    /// </summary>
    public static class StringExtensions
    {
        /// <summary>
        /// Parse the provided string as an integer, throwing <see cref="InvalidDataException"/> if parsing fails.
        /// </summary>
        /// <param name="value">The string to parse.</param>
        /// <param name="paramName">A parameter name used in the exception message if parsing fails.</param>
        /// <returns>The parsed integer value.</returns>
        public static int ReadAsInt(this string value, string paramName)
        {
            if (!int.TryParse(value, out var result))
                throw new InvalidDataException($"Invalid integer value for {paramName}.");
            return result;
        }

        /// <summary>
        /// Parse the provided string as a double, throwing <see cref="InvalidDataException"/> if parsing fails.
        /// </summary>
        /// <param name="value">The string to parse.</param>
        /// <param name="paramName">A parameter name used in the exception message if parsing fails.</param>
        /// <returns>The parsed double value.</returns>
        public static double ReadAsDouble(this string value, string paramName)
        {
            if (!double.TryParse(value, out var result))
                throw new InvalidDataException($"Invalid double value for {paramName}.");
            return result;
        }
    }
}
