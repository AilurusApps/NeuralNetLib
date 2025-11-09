using AilurusApps.NeuralNetLib;

namespace AilurusApps.NeuralNetLibTests
{
	public class ReLUFunctionTests
	{
		private ReLUFunction _relu;

		[SetUp]
		public void Setup()
		{
			_relu = ReLUFunction.Instance;
		}

		[Test]
		public void Invoke_ZeroInput_ReturnsZero()
		{
			double input = 0;
			
			double result = _relu.Invoke(input);
			
			Assert.That(result, Is.EqualTo(0.0));
		}

		[Test]
		public void Invoke_PositiveInput_ReturnsSameValue()
		{
			double input = 5.5;
			
			double result = _relu.Invoke(input);
			
			Assert.That(result, Is.EqualTo(5.5));
		}

		[Test]
		public void Invoke_NegativeInput_ReturnsZero()
		{
			double input = -3.7;
			
			double result = _relu.Invoke(input);
			
			Assert.That(result, Is.EqualTo(0.0));
		}

		[Test]
		public void Invoke_LargePositiveInput_ReturnsSameValue()
		{
			double input = 1000.0;
			
			double result = _relu.Invoke(input);
			
			Assert.That(result, Is.EqualTo(1000.0));
		}

		[Test]
		public void Invoke_LargeNegativeInput_ReturnsZero()
		{
			double input = -1000.0;
			
			double result = _relu.Invoke(input);
			
			Assert.That(result, Is.EqualTo(0.0));
		}

		[Test]
		public void Invoke_SmallPositiveInput_ReturnsSameValue()
		{
			double input = 0.001;
			
			double result = _relu.Invoke(input);
			
			Assert.That(result, Is.EqualTo(0.001));
		}

		[Test]
		public void Invoke_SmallNegativeInput_ReturnsZero()
		{
			double input = -0.001;
			
			double result = _relu.Invoke(input);
			
			Assert.That(result, Is.EqualTo(0.0));
		}

		[Test]
		public void GetDerivativeValue_PositiveActivation_ReturnsOne()
		{
			double activationOutput = 5.0;
			
			double result = _relu.GetDerivativeValue(activationOutput);
			
			Assert.That(result, Is.EqualTo(1.0));
		}

		[Test]
		public void GetDerivativeValue_ZeroActivation_ReturnsZero()
		{
			double activationOutput = 0.0;
			
			double result = _relu.GetDerivativeValue(activationOutput);
			
			Assert.That(result, Is.EqualTo(0.0));
		}

		[Test]
		public void GetDerivativeValue_SmallPositiveActivation_ReturnsOne()
		{
			double activationOutput = 0.001;
			
			double result = _relu.GetDerivativeValue(activationOutput);
			
			Assert.That(result, Is.EqualTo(1.0));
		}

		[Test]
		public void Instance_ReturnsSameInstance()
		{
			var instance1 = ReLUFunction.Instance;
			var instance2 = ReLUFunction.Instance;
			
			Assert.That(instance1, Is.SameAs(instance2));
		}

		[TestCase(-10.0, 0.0)]
		[TestCase(-5.0, 0.0)]
		[TestCase(-1.0, 0.0)]
		[TestCase(-0.5, 0.0)]
		[TestCase(0.0, 0.0)]
		[TestCase(0.5, 0.5)]
		[TestCase(1.0, 1.0)]
		[TestCase(5.0, 5.0)]
		[TestCase(10.0, 10.0)]
		public void Invoke_VariousInputs_ReturnsExpectedOutput(double input, double expected)
		{
			double result = _relu.Invoke(input);
			
			Assert.That(result, Is.EqualTo(expected));
		}

		[TestCase(0.001, 1.0)]
		[TestCase(0.5, 1.0)]
		[TestCase(1.0, 1.0)]
		[TestCase(5.0, 1.0)]
		[TestCase(10.0, 1.0)]
		public void GetDerivativeValue_VariousPositiveActivations_ReturnsOne(double activation, double expected)
		{
			double result = _relu.GetDerivativeValue(activation);
			
			Assert.That(result, Is.EqualTo(expected));
		}

		[Test]
		public void GetDerivativeValue_ZeroAndNegativeActivations_ReturnsZero()
		{
			// Note: In proper ReLU, negative activations shouldn't occur since ReLU output is always >= 0
			// But we test the derivative behavior at the boundary
			Assert.That(_relu.GetDerivativeValue(0.0), Is.EqualTo(0.0));
			Assert.That(_relu.GetDerivativeValue(-0.001), Is.EqualTo(0.0));
		}
	}
}
