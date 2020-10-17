#include <iostream>
#include <random>

#include "layer.h"

Layer::Layer(const unsigned int inputSize, const unsigned int outputSize, const unsigned int accumulatorBits, const double outScaling, const double stdDev):
    _inputSize(inputSize),
    _outputSize(outputSize),
    _output(outputSize),
    _stdDev(stdDev),
    _quantization(false),
    _accumulatorBits(accumulatorBits),
    _outScaling(outScaling)

{}

Layer::~Layer() {}

unsigned int Layer::getInputSize() const {
    return _inputSize;
}

unsigned int Layer::getOutputSize() const {
    return _outputSize;
}

double Layer::getOutput(unsigned int i) const {
    return _output.get(i);
}

const Input& Layer::output() const {return _output;}

void Layer::printOutput() const {
    _output.print();
}

void Layer::setQuantization(bool q) {
    _quantization = q;
}

