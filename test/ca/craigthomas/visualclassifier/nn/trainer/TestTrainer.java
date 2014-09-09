/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.nn.trainer;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

import ca.craigthomas.visualclassifier.nn.network.NeuralNetwork;
import ca.craigthomas.visualclassifier.nn.trainer.Trainer;

public class TestTrainer {

    private Trainer mTrainer;
    private List<Integer> mLayerSizes;

    @Test
    public void testTrainerLearnNOTFunction() {
        Random random = new Random();
        mLayerSizes = Arrays.asList(1, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix testInputs = DoubleMatrix.ones(10, 1);
        DoubleMatrix testOutputs = DoubleMatrix.ones(10, 1);
        
        for (int index = 0; index < 500; index++) {
            double value = (double)random.nextInt(100) + 1;
            if (value > 50.0) {
                inputs.put(index, 0, 0.0);
                outputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 1.0);
                outputs.put(index, 0, 0.0);                
            }
        }

        mTrainer = new Trainer.Builder(mLayerSizes, inputs, outputs)
                .learningRate(0.001).maxIterations(10000).heartBeat(0)
                .lambda(1.0).build();
        mTrainer.train();

        NeuralNetwork network = mTrainer.getNeuralNetwork();
        for (int index = 0; index < 10; index++) {
            double value = (double)random.nextInt(100) + 1;
            if (value > 50.0) {
                testInputs.put(index, 0, 1.0);
                testOutputs.put(index, 0, 0.0);
            } else {
                testInputs.put(index, 0, 0.0);
                testOutputs.put(index, 0, 1.0);
            }
        }
        DoubleMatrix predictions = network.predict(testInputs);
        Assert.assertArrayEquals(testOutputs.toArray(), predictions.toArray(), 0.15);
    }

    @Test
    public void testTrainerLearnANDFunction() {
        Random random = new Random();
        mLayerSizes = Arrays.asList(2, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 2);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix testInputs = DoubleMatrix.ones(10, 2);
        DoubleMatrix testOutputs = DoubleMatrix.ones(10, 1);
        
        for (int index = 0; index < 500; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                inputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                inputs.put(index, 1, 1.0);
            } else {
                inputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 && value2 > 50.0) {
                outputs.put(index, 0, 1.0);
            } else {
                outputs.put(index, 0, 0.0);
            }
        }

        mTrainer = new Trainer.Builder(mLayerSizes, inputs, outputs)
                .learningRate(0.001).maxIterations(20000).heartBeat(0)
                .lambda(1.0).build();
        mTrainer.train();

        NeuralNetwork network = mTrainer.getNeuralNetwork();
        for (int index = 0; index < 10; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                testInputs.put(index, 0, 1.0);
            } else {
                testInputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                testInputs.put(index, 1, 1.0);
            } else {
                testInputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 && value2 > 50.0) {
                testOutputs.put(index, 0, 1.0);
            } else {
                testOutputs.put(index, 0, 0.0);
            }
        }
        DoubleMatrix predictions = network.predict(testInputs);
        Assert.assertArrayEquals(testOutputs.toArray(), predictions.toArray(), 0.15);
    }
    
    @Test
    public void testTrainerLearnORFunction() {
        Random random = new Random();
        mLayerSizes = Arrays.asList(2, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 2);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix testInputs = DoubleMatrix.ones(10, 2);
        DoubleMatrix testOutputs = DoubleMatrix.ones(10, 1);
        
        for (int index = 0; index < 500; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                inputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                inputs.put(index, 1, 1.0);
            } else {
                inputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 || value2 > 50.0) {
                outputs.put(index, 0, 1.0);
            } else {
                outputs.put(index, 0, 0.0);
            }
        }

        mTrainer = new Trainer.Builder(mLayerSizes, inputs, outputs)
                .learningRate(0.001).maxIterations(15000).heartBeat(0)
                .lambda(1.0).build();
        mTrainer.train();

        NeuralNetwork network = mTrainer.getNeuralNetwork();
        for (int index = 0; index < 10; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                testInputs.put(index, 0, 1.0);
            } else {
                testInputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                testInputs.put(index, 1, 1.0);
            } else {
                testInputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 || value2 > 50.0) {
                testOutputs.put(index, 0, 1.0);
            } else {
                testOutputs.put(index, 0, 0.0);
            }
        }
        DoubleMatrix predictions = network.predict(testInputs);
        Assert.assertArrayEquals(testOutputs.toArray(), predictions.toArray(), 0.15);
    }
}
