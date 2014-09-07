/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.nn.activation;

import org.jblas.DoubleMatrix;

/**
 * An interface to capture an activation function. There are several different
 * activation functions that one might wish to use within a neural network. At
 * a minimum, each activation function can be applied to a particular matrix,
 * as well as return the gradient for that function along a given input.
 * 
 * @author thomas
 */
public interface IActivationFunction {

    public DoubleMatrix apply(DoubleMatrix input);
    
    public DoubleMatrix gradient(DoubleMatrix input);
    
    public double apply(double input);
}
