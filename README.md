# NeuralNetLib
A simple Artificial Neural Network class library for C# .NET

<p align="center">
<a href="https://github.com/AilurusApps/NeuralNetLib/blob/main/LICENSE">
<img src="https://img.shields.io/badge/License-GPL--3.0-blue.svg" alt="License: GPL-3.0" />
</a>
<img src="https://img.shields.io/github/languages/top/AilurusApps/NeuralNetLib?color=9b1b9e" alt="Top Language: C#" />
<img src="https://img.shields.io/github/last-commit/AilurusApps/NeuralNetLib?color=green" alt="Last Commit" />
</p>


**About The Project**

NeuralNetLib is a lightweight and straightforward class library designed to allow C# developers to easily implement and experiment with Artificial Neural Networks (ANNs) directly within their .NET projects.

The goal is to provide a core, performant, and easily understandable implementation of fundamental neural network concepts without the complexity of larger, full-featured machine learning frameworks. It is ideal for educational purposes, small-scale projects, and developers who want a deep understanding of how ANNs work under the hood.

Built entirely in C# for the .NET ecosystem.

**Getting Started**

To get a local copy up and running, follow these simple steps.

**Installation**

The library is designed to be easily included in any compatible C# project.

1. NuGet Package

*As of writing, no packages have been published for this library. Please include via DLL or manual source inclusion.

2. Manual Source Code Inclusion

Alternatively, you can include the source files directly into your project:

Clone the repository:

Bash

git clone https://github.com/AilurusApps/NeuralNetLib.git

Reference the NeuralNetLib.csproj file or include the necessary C# files in your solution.

**Usage**

Once installed, you can begin defining, training, and running your neural networks.

Here is a conceptual example of a simple Multilayer Perceptron (MLP) for XOR logic:

C#
```
using AilurusApps.NeuralNetLib;
using System;

public class XorExample
{
    public static void Main()
    {
        // 1. Define the network structure: 2 inputs, 3 hidden neurons, 1 output
        var network = NeuralNetworkFactory.Build(2, 1, [3]);

        // 2. Define the training data (XOR Logic)
        double[][] inputs = new double[][]
        {
          new[] { 0.0, 0.0 }, // Expected: 0
          new[] { 0.0, 1.0 }, // Expected: 1
          new[] { 1.0, 0.0 }, // Expected: 1
          new[] { 1.0, 1.0 }  // Expected: 0
        };

        double[][] targets = new double[][]
        {
          new[] { 0.0 },
          new[] { 1.0 },
          new[] { 1.0 },
          new[] { 0.0 }
        };

        // 3. Train the network (using parameters like learning rate, momentum)
        var algorithm = new Backpropagation()
        {
            LearningRate = 0.2,
            Momentum = 0.1
        };

        var trainingData = inputs.Zip(targets).ToDictionary(t => (t.First[0], t.First[1]), t => new TrainingData(t.First, t.Second));

        var trainer = new Trainer<(double,double)>(trainingData, algorithm);

        trainer.Retrain(network, tolerance: 0, maxIterations: 10000);

        // 4. Test the network
        Console.WriteLine("--- Testing Results ---");
        foreach (var input in inputs)
        {
            network.Fire(input);
            var output = network.Outputs[0].Value;
            Console.WriteLine($"Input: ({input[0]}, {input[1]}) -> Output: {output:F4}");
        }
    }
}
```

Note: Refer to the /src folder for detailed class and method documentation.

**Features**

NeuralNetLib aims to provide a robust and extensible foundation for neural network development. Key features include:

Multilayer Perceptron (MLP) Support: Easily create fully-connected feed-forward networks.

Backpropagation Training: Standard implementation of the backpropagation algorithm for efficient learning.

Customizable Activation Functions: Support for popular activation functions (e.g., Sigmoid, HyperTan).

Simple Data Structures: Clear and accessible internal structures to manage weights, biases, and layers.

Pure C# Implementation: No external complex dependencies required, ensuring simplicity and ease of integration.

**Roadmap**

[ ] Implement additional standard activation functions (e.g., Leaky ReLU).

[ ] Integrate different optimization algorithms (e.g., Momentum, Adam).

[ ] Publish a stable NuGet package.

**Contributing**

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Fork the Project.

Create your Feature Branch (git checkout -b feature/AmazingFeature).

Commit your Changes (git commit -m 'Add some AmazingFeature').

Push to the Branch (git push origin feature/AmazingFeature).

Open a Pull Request.

**License**

Distributed under the GPL-3.0 License. See LICENSE for more information.

**Contact**

AilurusApps - https://ailurusapps.com

Project Link: https://github.com/AilurusApps/NeuralNetLib
