/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.neuralnetwork;

import static org.junit.Assert.assertEquals;

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
    
    @Test
    public void testNeuralNetworkGetCostNoRegularization() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });
        
        double expectedCost = 0.88101;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).expectedValues(expectedOutputs).build();
        mNeuralNetwork.predict(testInputs);

        assertEquals(expectedCost, mNeuralNetwork.getCostNoRegularization(8), 0.0005);
    }
    
    @Test
    public void testGetMatrixNoBiasRemovesBiasUnit() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {1.000, 0.69149, 0.23113}
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.69149, 0.23113}     
        });

        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.getMatrixNoBias(theta1);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void getThetaRegularizationReturnsZeroWithDefaultLambda() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {1.000, 0.69149, 0.23113}
        });
        
        double expected = 0.0;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        double result = mNeuralNetwork.getThetaRegularization(1);
        
        assertEquals(expected, result, 0.0001);
    } 
    
    @Test
    public void getThetaRegularizationReturnsCorrectWithLambdaOne() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {1.000, 0.69149, 0.23113}
        });
        
        double expected = 0.26579;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).lambda(1.0).build();
        double result = mNeuralNetwork.getThetaRegularization(1);
        
        assertEquals(expected, result, 0.0001);
    }
    
    @Test
    public void testNeuralNetworkGetCostNoLambda() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });

        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });

        double expectedCost = 0.88101;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas)
                .inputs(testInputs).expectedValues(expectedOutputs).build();
        mNeuralNetwork.predict(testInputs);

        assertEquals(expectedCost, mNeuralNetwork.getCost(), 0.0005);
    }
    
    @Test
    public void testNeuralNetworkGetCostWithLambda() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });

        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });
        
        double lambda = 1.0;

        double expectedCost = 0.91423;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas)
                .inputs(testInputs).expectedValues(expectedOutputs)
                .lambda(lambda).build();
        mNeuralNetwork.predict(testInputs);

        assertEquals(expectedCost, mNeuralNetwork.getCost(), 0.0005);
    }
}
