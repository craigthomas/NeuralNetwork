/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.activation;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * A class used to calculate the hyperbolic tangent of all of the 
 * elements in a DoubleMatrix. The HyperbolicTangent function is simply:
 * 
 *   TanH(t) =  1 - e^(-2t)
 *             -------------
 *              1 + e^(-2t)
 *            
 * @author thomas
 */
public class HyperbolicTangent implements IActivationFunction {

    public HyperbolicTangent() {
    }
    
    /**
     * Calculate the HyperbolicTangent value for every element in the specified 
     * matrix.
     * 
     * @param input the DoubleMatrix to use as input
     * @return the tanh value of the input matrix
     */
    public DoubleMatrix apply(DoubleMatrix input) {
        DoubleMatrix result = new DoubleMatrix().copy(input).muli(2.0);
        MatrixFunctions.expi(result);
        return result.sub(1.0).div(result.add(1.0));
    }

    /**
     * Calculate the Sigmoid value for a single double.
     * 
     * @param input the double to use as input
     * @return the sigmoid value of the input
     */
    public double apply(double input) {
        return (1.0 - Math.exp(-2*input)) / (1.0 + Math.exp(-2*input));
    }
}
