/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * A static class used to calculate the sigmoid of all of the elements in a
 * DoubleMatrix. The Sigmoid function is simply:
 * 
 *   S(t) =       1
 *           ------------
 *            1 + e^(-t)
 *            
 * @author thomas
 */
public class Sigmoid {

    /**
     * Make the constructor static so that it cannot be instantiated. 
     */
    private Sigmoid() {
    }
    
    /**
     * Calculate the Sigmoid value for every element in the specified matrix.
     * 
     * @param input the DoubleMatrix to use as input
     * @return the sigmoid value of the input matrix
     */
    public static DoubleMatrix apply(DoubleMatrix input) {
        DoubleMatrix result = new DoubleMatrix().copy(input);
        DoubleMatrix ones = DoubleMatrix.ones(result.rows, result.columns);
        result.muli(-1);
        MatrixFunctions.expi(result);
        result.addi(1);
        return ones.divi(result);
    }

    /**
     * Calculate the Sigmoid value for a single double.
     * 
     * @param input the double to use as input
     * @return the sigmoid value of the input
     */
    public static double apply(double input) {
        return (1.0 / (1.0 + Math.exp(-input)));
    }
}
