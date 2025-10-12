using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
    public class SigmoidFunctionTests
    {
        private SigmoidFunction _sigmoid;

        [SetUp]
        public void Setup()
        {
            _sigmoid = SigmoidFunction.Instance;
        }

        [Test]
        public void Invoke_ZeroInput_ReturnsPointFive()
        {
            double input = 0;
            
            double result = _sigmoid.Invoke(input);
            
            Assert.That(result, Is.EqualTo(0.5).Within(1e-10));
        }

        [Test]
        public void Invoke_LargePositiveInput_ApproachesOne()
        {
            double input = 10;
            
            double result = _sigmoid.Invoke(input);
            
            Assert.That(result, Is.GreaterThan(0.99));
        }

        [Test]
        public void Invoke_LargeNegativeInput_ApproachesZero()
        {
            double input = -10;
            
            double result = _sigmoid.Invoke(input);
            
            Assert.That(result, Is.LessThan(0.01));
        }

        [Test]
        public void GetDerivativeValue_AtPointFive_ReturnsPointTwoFive()
        {
            double input = 0.5;
            
            double result = _sigmoid.GetDerivativeValue(input);
            
            Assert.That(result, Is.EqualTo(0.25).Within(1e-10));
        }
    }
}