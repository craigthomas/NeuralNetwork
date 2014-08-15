/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.neuralnetwork;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import ca.craigthomas.visualclassifier.neuralnetwork.Sigmoid;

public class TestSigmoid {

    @Test
    public void testSigmoidSingleValueZero() {
        assertEquals(0.5, Sigmoid.apply(0), 0.00001);
    }

    @Test
    public void testSigmoidSingleValueOne() {
        assertEquals(0.73106, Sigmoid.apply(1), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueFive() {
        assertEquals(0.99331, Sigmoid.apply(5), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueOneHundred() {
        assertEquals(1.0, Sigmoid.apply(100), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueNegativeOne() {
        assertEquals(0.26894, Sigmoid.apply(-1), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueNegativeFive() {
        assertEquals(0.0066929, Sigmoid.apply(-5), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueNegativeOneHundred() {
        assertEquals(0.0, Sigmoid.apply(-100), 0.00001);
    }

    @Test
    public void testSigmoidMatrixSingleRowColumnZero() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 0.0);
        DoubleMatrix result = Sigmoid.apply(doubleMatrix);
        assertEquals(0.5, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnOne() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 1.0);
        DoubleMatrix result = Sigmoid.apply(doubleMatrix);
        assertEquals(0.73106, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnFive() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 5.0);
        DoubleMatrix result = Sigmoid.apply(doubleMatrix);
        assertEquals(0.99331, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnOneHundred() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 100.0);
        DoubleMatrix result = Sigmoid.apply(doubleMatrix);
        assertEquals(1.0, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnNegativeOne() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -1.0);
        DoubleMatrix result = Sigmoid.apply(doubleMatrix);
        assertEquals(0.26894, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnNegativeFive() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -5.0);
        DoubleMatrix result = Sigmoid.apply(doubleMatrix);
        assertEquals(0.0066929, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnNegativeOneHundred() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -100.0);
        DoubleMatrix result = Sigmoid.apply(doubleMatrix);
        assertEquals(0.0, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixMultipleRowsAndColumns() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(new double[][] {{1.0, 5.0, 100.0}, {-1.0, -5.0, -100.0}});
        DoubleMatrix expected = new DoubleMatrix(new double[][] {{0.73106, 0.99331, 1.0}, {0.26894, 0.0066929, 0.0}});
        DoubleMatrix result = Sigmoid.apply(doubleMatrix);
        for (int row = 0; row < result.rows; row++) {
            for (int col = 0; col < result.columns; col++) {
                assertEquals(expected.get(row, col), result.get(row, col), 0.00001);
            }
        }
    }
}
