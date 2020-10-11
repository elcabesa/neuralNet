#include "activation.h"
#include "relu.h"
#include "linear.h"

Activation::Activation() {}

Activation::~Activation() {}

std::unique_ptr<Activation> ActivationFactory::create( const Activation::type t )
{
	if( t == Activation::type::linear)
	{
		return std::make_unique<linearActivation>();
	}
	else if(t == Activation::type::relu)
	{
		return std::make_unique<reluActivation>();
	}
	return std::make_unique<linearActivation>();
}

