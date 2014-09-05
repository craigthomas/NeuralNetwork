/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.neuralnetwork;

import java.util.List;
import java.lang.IllegalArgumentException;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.*;
import org.jblas.util.Random;

import ca.craigthomas.visualclassifier.activation.IActivationFunction;
import ca.craigthomas.visualclassifier.activation.Sigmoid;

/**
 * Implements a neural network. Multiple layers can be specified in the 
 * neural network by setting the initial layerSizes value to a list of 
 * the number of nodes desired in each layer. At a minimum, a neural network 
 * can consist of the layer sizes, and the theta values for the weights between
 * each layer. The predict function can then be used to calculate the resulting
 * values for a given set of inputs.
 * 
 * @author thomas
 */
public class NeuralNetwork {

    private int[] mLayerSizes;
    private DoubleMatrix[] mThetas;
    private final IActivationFunction mActivationFunction;
    private DoubleMatrix[] mActivations;
    private Random mRandom;
    private DoubleMatrix mIdentities;
    private final double mLambda;
    
    public static class Builder {
        
        private int[] mLayerSizes;
        private DoubleMatrix[] mThetas;
        private DoubleMatrix mInputs;
        private DoubleMatrix mExpected;
        private double mLambda;
        private boolean mTrain;
        private IActivationFunction mActivationFunction;
        
        /**
         * Constructs a neural network with the specified layers and their
         * sizes. For example, a neural network that has layerSizes equal to
         * [5, 4, 1], will build a neural network with 3 layers. The first layer
         * will have 5 nodes, and the second layer will have 4 nodes, and the
         * third layer will have 1 node. The first and last layer in the list
         * are assumed to be the input layer and output layer, respectively.
         *   
         * @param layerSizes A list of the number of nodes in each layer 
         */
        public Builder(List<Integer> layerSizes) {
            if (layerSizes.size() < 2) {
                throw new IllegalArgumentException("must have at least 2 layers");
            }
            mLayerSizes = new int[layerSizes.size()];
            for (int index = 0; index < layerSizes.size(); index++) {
                mLayerSizes[index] = layerSizes.get(index).intValue();
            }
            mTrain = false;
            mLambda = 0.0;
        }
        
        /**
         * Sets the theta values (weights) to be used between the various 
         * layers in the neural network. 
         * 
         * @param thetas the list of theta values to use in the network
         * @return the builder for the neural network
         */
        public Builder theta(List<DoubleMatrix> thetas) {
            mThetas = thetas.toArray(new DoubleMatrix[thetas.size()]);
            return this;
        }
        
        /**
         * The lambda parameter controls how regularization of the neural 
         * network operates. It is used as a training parameter to help 
         * prevent over and under fitting. The default value of 0 means that
         * no regularization will occur.
         * 
         * @param lambda the regularization parameter
         * @return the builder for the neural network
         */
        public Builder lambda(double lambda) {
            mLambda = lambda;
            return this;
        }
        
        /**
         * Sets the training examples to use to train the network. 
         * 
         * @param inputs the training examples to use for training the network
         * @return the builder for the neural network
         */
        public Builder inputs(DoubleMatrix inputs) {
            mInputs = inputs;
            return this;
        }
        
        /**
         * Sets the actual observed outputs for training inputs. 
         * 
         * @param outputs the true valued output for the training examples
         * @return the builder for the neural network
         */
        public Builder expectedValues(DoubleMatrix expected) {
            mExpected = expected;
            return this;
        }
        
        /**
         * Notifies the builder that it should train the neural network on the 
         * inputs and outputs provided when the neural network is built. The
         * default training parameter is false, meaning it will simply 
         * construct the neural network with the thetas provided. A value of
         * true will instruct the builder to run forward propagation followed
         * by back propagation on the specified inputs and outputs. It will
         * also calculate the training cost and gradient for each of the theta
         * layers
         * 
         * @return the builder for the neural network
         */
        public Builder train() {
            mTrain = !mTrain;
            return this;
        }
        
        /**
         * Sets the activation function for the neural network. If none is 
         * specified, the Sigmoid activation function is used by default.
         * 
         * @param activationFunction the activation function to use
         * @return the builder for the neural network
         */
        public Builder activationFunction(IActivationFunction activationFunction) {
            mActivationFunction = activationFunction;
            return this;
        }
        
        /**
         * The builder for the NeuralNetwork.
         * 
         * @return the NeuralNetwork built by the builder
         */
        public NeuralNetwork build() {
            if (mActivationFunction == null) {
                mActivationFunction = new Sigmoid();
            }
            return new NeuralNetwork(this);
        }
    }
    
    /**
     * Build the neural network based on the values set by the builder.
     * 
     * @param builder the builder with instructions on building the network
     */
    private NeuralNetwork(Builder builder) {
        mLayerSizes = builder.mLayerSizes;
        mThetas = builder.mThetas;
        mActivationFunction = builder.mActivationFunction;
        mActivations = new DoubleMatrix[mLayerSizes.length];
        mIdentities = builder.mExpected;
        mLambda = builder.mLambda;
        
        if ((builder.mTrain) && (builder.mInputs == null)) {
            throw new IllegalStateException("Cannot train network without training examples");
        } else if (builder.mTrain) {
            setInputs(builder.mInputs);
        }
        
        // TODO: perform random initialization of thetas for symmetry breaking
        // TODO: 
    }
    
    /**
     * Will initialize various parameters of the network.
     */
    protected void initializeNetwork() {
    }
    
    /**
     * Adds a bias unit on to the specified matrix. This essentially adds a 
     * column of 1's to the input matrix.
     * 
     * @param input the matrix to add on to
     * @return a new matrix with a bias unit
     */
    public DoubleMatrix addBias(DoubleMatrix input) {
        int length = input.rows;
        DoubleMatrix ones = DoubleMatrix.ones(length, 1);
        return DoubleMatrix.concatHorizontally(ones, input.dup());
    }
    
    /**
     * Sets the inputs for the neural network. Will add a bias unit to 
     * the set of inputs.
     * 
     * @param input the matrix to treat as input
     */
    public void setInputs(DoubleMatrix input) {
        mActivations[0] = addBias(input);
    }
    
    /**
     * Apply forward propagation to the neural network, updating the activations
     * as it moves through the network.
     */
    protected void forwardPropagation() {
        for (int index = 0; index < mActivations.length - 1; index++) {
            DoubleMatrix theta = mThetas[index];
            DoubleMatrix activation = mActivations[index];
            DoubleMatrix z = activation.mmul(theta.transpose());
            mActivations[index+1] = mActivationFunction.apply(z);
            if (index+1 != mActivations.length - 1) {
                mActivations[index+1] = addBias(mActivations[index+1]);
            } 
        }
    }
    
    /**
     * Returns a the specified matrix without a bias unit attached (i.e. the
     * matrix without the first column of 1's).
     * 
     * @param matrix the matrix to slice
     * @return the matrix without the first column
     */
    protected DoubleMatrix getMatrixNoBias(DoubleMatrix matrix) {
        int numRows = matrix.rows;
        int numCols = matrix.columns;
        Range rows = new IntervalRange(0, numRows);
        Range cols = new IntervalRange(1, numCols);
        return matrix.get(rows, cols);
    }
    
    /**
     * Return the specified theta matrix.
     * 
     * @param thetaNum the theta number to return
     * @return the theta values for that layer
     */
    public DoubleMatrix getTheta(int thetaNum) {
        return mThetas[thetaNum];
    }
    
    /**
     * Computes the regularization term for the thetas in the network.
     * 
     * @param numInputs the number of inputs over which to regularize
     * @return the regularized term for the thetas
     */
    public double getThetaRegularization(int numInputs) {
        double thetaSum = 0.0;
        for (DoubleMatrix theta : mThetas) {
            DoubleMatrix thetaNoBias = getMatrixNoBias(theta);
            thetaSum += thetaNoBias.mul(thetaNoBias).sum();
        }
        return (mLambda / (2*numInputs)) * thetaSum;
    }

    /**
     * Get the cost associated with the current thetas.
     * 
     * @return the cost of the thetas
     */
    public double getCostNoRegularization(int numInputs) {
        DoubleMatrix outputLayer = mActivations[mActivations.length - 1];
        DoubleMatrix expected = mIdentities.transpose().mul(-1);
        DoubleMatrix posTerm = expected.mmul(MatrixFunctions.log(outputLayer));
        DoubleMatrix negTerm = expected.add(1.0).mmul(MatrixFunctions.log(outputLayer.mul(-1).add(1.0)));
        return (1.0/numInputs) * posTerm.sub(negTerm).sum();
    }
    
    /**
     * Calculates cost with regularization. Regularization will not be applied
     * when the lambda value is 0 (by default).
     * 
     * @return the cost
     */
    public double getCost() {
        int numInputs = mActivations[0].rows;
        return getCostNoRegularization(numInputs) + getThetaRegularization(numInputs);
    }
    
    /**
     * Given a trained neural network (i.e. a trained network that has learned
     * the values for theta, or has pre-supplied values for theta, plus the 
     * activations layers), compute the resulting output values given some data 
     * as input. 
     * 
     * @param data the examples to predict
     * @return the predicted values (classes)
     */
    public DoubleMatrix predict(DoubleMatrix data) {
        setInputs(data);
        forwardPropagation();
        return mActivations[mLayerSizes.length - 1];
    }
    
}
