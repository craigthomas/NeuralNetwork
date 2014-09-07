package ca.craigthomas.visualclassifier.activation;

import org.jblas.DoubleMatrix;

public interface IActivationFunction {

    public DoubleMatrix apply(DoubleMatrix input);
    
    public DoubleMatrix gradient(DoubleMatrix input);
    
    public double apply(double input);
}
