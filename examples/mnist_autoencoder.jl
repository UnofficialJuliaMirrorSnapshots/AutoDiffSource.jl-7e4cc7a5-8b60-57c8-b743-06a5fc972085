# based on http://int8.io/automatic-differentiation-machine-learning-julia/

using MNIST # if not installed try Pkg.add("MNIST")
using AutoDiffSource # if not installed try Pkg.add("AutoDiffSource")
using PyPlot # if not installed try Pkg.add("PyPlot")

sigmoid(x) = 1 / (1 + exp(x))

function autoencoder(We1, We2, Wd, b1, b2, input)
    firstLayer = sigmoid.(We1 * input + b1)
    encodedInput = sigmoid.(We2 * firstLayer + b2)
    reconstructedInput = sigmoid.(Wd * encodedInput)
    return reconstructedInput
end

@δ function autoencoderError(We1, We2, Wd, b1, b2, input)
    reconstructedInput = autoencoder(We1, We2, Wd, b1, b2, input)
    return sum((input - reconstructedInput).^2)
end

@assert checkdiff(autoencoderError, δautoencoderError, randn(3,3), randn(3,3), rand(3,3), randn(3), randn(3), randn(3))

function initializeNetworkParams(inputSize, layer1Size, layer2Size)
    We1 =  1f-1 * randn(Float32, layer1Size, inputSize)
    b1 = zeros(Float32, layer1Size)
    We2 =  1f-1 * randn(Float32, layer2Size, layer1Size)
    b2 = zeros(Float32, layer2Size)
    Wd = 1f-1 * randn(Float32, inputSize, layer2Size)
    return (We1, We2, b1, b2, Wd)
end

function show_digits(testing, We1, We2, b1, b2, Wd)
    clf()
    total_error = 0
    for l = 1:12
        input = testing[:, rand(1:size(testing, 2))]
        reconstructedInput = autoencoder(We1, We2, Wd, b1, b2, input)
        subplot(4, 6, l*2-1)
        pcolor(rotl90(reshape(input, 28, 28)'); cmap="Greys")
        subplot(4, 6, l*2)
        pcolor(rotl90(reshape(reconstructedInput, 28, 28)'); cmap="Greys")
        total_error += sum((input - reconstructedInput).^2)
    end
    tight_layout()
    total_error / 12
end

function trainAutoencoder(epochs, training, testing, We1, We2, b1, b2, Wd, alpha)
    for k in 1:epochs
        total_error = 0.
        for i in 1:size(training, 2)
            input = training[:,i]
            val, ∇autoencoderError = δautoencoderError_const6(We1, We2, Wd, b1, b2, input)
            total_error += val
            if mod(i, 10_000) == 0
                test_error = show_digits(testing, We1, We2, b1, b2, Wd)
                @printf("epoch=%d iter=%d train_error=%.2f test_error=%.2f\n", k, i, total_error/10_000, test_error)
                total_error = 0.
            end
            ∂We1, ∂We2, ∂Wd, ∂b1, ∂b2 = ∇autoencoderError()
            We1 -= alpha * ∂We1
            We2 -= alpha * ∂We2
            Wd  -= alpha * ∂Wd
            b1  -= alpha * ∂b1
            b2  -= alpha * ∂b2
        end
    end
    return (We1, We2, b1, b2, Wd)
end

# read input MNIST data
training = convert(Matrix{Float32}, MNIST.traindata()[1] / 255)
testing = convert(Matrix{Float32}, MNIST.testdata()[1] / 255)

# 784 -> 300 -> 100 -> 784 with weights normally distributed (with small variance)
We1, We2, b1, b2, Wd = initializeNetworkParams(784, 300, 100)

# 4 epochs with alpha = 0.02
@time We1, We2, b1, b2, Wd = trainAutoencoder(4, training, testing, We1, We2, b1, b2, Wd, 2f-2)
