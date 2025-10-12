using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
    public class HyperTanFunctionTests
    {
        private HyperTanFunction _hyperTan;

        [SetUp]
        public void Setup()
        {
            _hyperTan = HyperTanFunction.Instance;
        }

        [Test]
        public void Invoke_ZeroInput_ReturnsZero()
        {
            double input = 0;
            
            double result = _hyperTan.Invoke(input);
            
            Assert.That(result, Is.EqualTo(0).Within(1e-10));
        }

        [Test]
        public void Invoke_LargePositiveInput_ApproachesOne()
        {
            double input = 10;
            
            double result = _hyperTan.Invoke(input);
            
            Assert.That(result, Is.GreaterThan(0.99));
        }

        [Test]
        public void Invoke_LargeNegativeInput_ApproachesNegativeOne()
        {
            double input = -10;
            
            double result = _hyperTan.Invoke(input);
            
            Assert.That(result, Is.LessThan(-0.99));
        }

        [Test]
        public void GetDerivativeValue_AtZero_ReturnsOne()
        {
            double input = 0;
            
            double result = _hyperTan.GetDerivativeValue(input);
            
            Assert.That(result, Is.EqualTo(1).Within(1e-10));
        }
    }
}