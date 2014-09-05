/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.neuralnetwork;

import java.lang.IllegalArgumentException;
import java.util.Arrays;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import ca.craigthomas.visualclassifier.neuralnetwork.NeuralNetwork;

public class TestNeuralNetwork {

    private NeuralNetwork mNeuralNetwork;
    private List<Integer> layerSizes;
    
    @Before
    public void setUp() {
        mNeuralNetwork = null;
    }
    
    @Test (expected=IllegalArgumentException.class)
    public void testLessThanTwoLayers() {
        layerSizes = Arrays.asList(3);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).build();
    }
    
    @Test
    public void testPredictANDFunction() {
        layerSizes = Arrays.asList(3, 1);
        DoubleMatrix theta = new DoubleMatrix(new double [][] {
                {-300.0, 200.0, 200.0}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0, 0.0},  // x1 = 0, x2 = 0
                {0.0, 1.0},  // x1 = 0, x2 = 1
                {1.0, 0.0},  // x1 = 1, x2 = 0
                {1.0, 1.0}   // x1 = 1, x2 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0},       // x1 (0) and x2 (0) == 0
                {0.0},       // x1 (0) and x2 (1) == 0
                {0.0},       // x1 (1) and x2 (0) == 0
                {1.0}        // x1 (1) and x2 (1) == 0
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testPredictORFunction() {
        layerSizes = Arrays.asList(3, 1);
        DoubleMatrix theta = new DoubleMatrix(new double [][] {
                {-100.0, 200.0, 200.0}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0, 0.0},  // x1 = 0, x2 = 0
                {0.0, 1.0},  // x1 = 0, x2 = 1
                {1.0, 0.0},  // x1 = 1, x2 = 0
                {1.0, 1.0}   // x1 = 1, x2 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0},       // x1 (0) or x2 (0) == 0
                {1.0},       // x1 (0) or x2 (1) == 1
                {1.0},       // x1 (1) or x2 (0) == 1
                {1.0}        // x1 (1) or x2 (1) == 1
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testPredictXORFunction() {
        layerSizes = Arrays.asList(3, 1, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {-100.0, 200.0, 200.0},
                {200.0, -150.0, -150.0}
        });
        
        DoubleMatrix theta2 = new DoubleMatrix(new double [][] {
                {-300.0, 200.0, 200.0}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0, 0.0},  // x1 = 0, x2 = 0
                {0.0, 1.0},  // x1 = 0, x2 = 1
                {1.0, 0.0},  // x1 = 1, x2 = 0
                {1.0, 1.0}   // x1 = 1, x2 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0},       // x1 (0) xor x2 (0) == 0
                {1.0},       // x1 (0) xor x2 (1) == 1
                {1.0},       // x1 (1) xor x2 (0) == 1
                {0.0}        // x1 (1) xor x2 (1) == 0
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1, theta2);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testPredictNOTFunction() {
        layerSizes = Arrays.asList(3, 1);
        DoubleMatrix theta = new DoubleMatrix(new double [][] {
                {100.0, -200.0}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0},  // x1 = 0
                {1.0}   // x1 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {1.0},       // not x1 (0) == 1
                {0.0},       // not x1 (1) == 0
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testPredictXNORFunction() {
        layerSizes = Arrays.asList(3, 2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {-300.0, 200.0, 200.0},
                {100.0, -200.0, -200.0},
        });
        
        DoubleMatrix theta2 = new DoubleMatrix(new double [][] {
                {-100.0, 200.0, 200.0},
        }); 
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0, 0.0},  // x1 = 0, x2 = 0
                {0.0, 1.0},  // x1 = 0, x2 = 1
                {1.0, 0.0},  // x1 = 1, x2 = 0
                {1.0, 1.0}   // x1 = 1, x2 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {1.0},       // x1 (0) xnor x2 (0) == 1
                {0.0},       // x1 (0) xnor x2 (1) == 0
                {0.0},       // x1 (1) xnor x2 (0) == 0
                {1.0}        // x1 (1) xnor x2 (1) == 1
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1, theta2);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
}
