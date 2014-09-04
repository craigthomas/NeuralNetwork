/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.neuralnetwork;

import java.util.List;
import java.lang.IllegalArgumentException;

import org.jblas.DoubleMatrix;

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

    private final List<Integer> mLayerSizes;
    private final List<DoubleMatrix> mThetas;
    private final IActivationFunction mActivationFunction;
    
    public static class Builder {
        
        private List<Integer> mLayerSizes;
        private List<DoubleMatrix> mThetas;
        private DoubleMatrix mInputs;
        private DoubleMatrix mOutputs;
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
            mLayerSizes = layerSizes;
            mTrain = false;
        }
        
        /**
         * Sets the theta values (weights) to be used between the various 
         * layers in the neural network. 
         * 
         * @param thetas the list of theta values to use in the network
         * @return the builder for the neural network
         */
        public Builder theta(List<DoubleMatrix> thetas) {
            mThetas = thetas;
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
        public Builder outputs(DoubleMatrix outputs) {
            mOutputs = outputs;
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
            return new NeuralNetwork(this);
        }
    }
    
    private NeuralNetwork(Builder builder) {
        mLayerSizes = builder.mLayerSizes;
        mThetas = builder.mThetas;
        mActivationFunction = builder.mActivationFunction;
    }
    
    /**
     * Return the specified theta matrix.
     * 
     * @param thetaNum the theta number to return
     * @return the theta values for that layer
     */
    public DoubleMatrix getTheta(int thetaNum) {
        return mThetas.get(thetaNum);
    }
    
    /**
     * Given a trained neural network (i.e. a trained network that has learned
     * the values for theta, or has pre-supplied values for theta), compute
     * the resulting output values given some data as input. 
     * 
     * @param data the examples to predict
     * @return the predicted values (classes)
     */
    public DoubleMatrix predict(DoubleMatrix data) {
        int length = data.rows;
        DoubleMatrix lastLayer = data.dup();
        
        for (int layer = 0; layer < mThetas.size(); layer++) {
            DoubleMatrix ones = DoubleMatrix.ones(length, 1);
            DoubleMatrix activation = DoubleMatrix.concatHorizontally(ones, lastLayer);
            lastLayer = mActivationFunction.apply(activation.mmul(mThetas.get(layer).transpose()));
        }
        
        return lastLayer;
    }
    
}
