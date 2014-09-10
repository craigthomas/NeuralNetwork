/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.dataset;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

public class TestDataSet {

    private static final String BAD_FILENAME = "/this_file_does_not_exist.csv";
    private static final String SAMPLE_FILE = "resources/small_dataset_example.csv";

    private DataSet mDataSet;
    
    @Test (expected=IOException.class)
    public void testReadFromNonExistentCSV() throws IOException {
        mDataSet = new DataSet(true);
        mDataSet.addFromCSVFile(BAD_FILENAME);
    }
    
    @Test
    public void newDataSetIsEmpty() {
        mDataSet = new DataSet(false);
        assertTrue(mDataSet.getSamples() == null);
        assertTrue(mDataSet.getTruth() == null);
        assertEquals(0, mDataSet.getNumColsSamples());
        assertEquals(0, mDataSet.getNumColsTruth());
        assertEquals(0, mDataSet.getNumSamples());
    }
    
    @Test
    public void testReadFromCSVFileNoTruthWorksCorrectly() throws IOException {
        mDataSet = new DataSet(false);
        mDataSet.addFromCSVFile(SAMPLE_FILE);
        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {1.0, 1.0, 1.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0}
        });

        assertFalse(mDataSet.hasTruth());
        assertEquals(0, mDataSet.getNumColsTruth());
        assertEquals(3, mDataSet.getNumColsSamples());
        assertEquals(4, mDataSet.getNumSamples());
        assertTrue(mDataSet.getTruth() == null);
        Assert.assertArrayEquals(expected.toArray(), mDataSet.getSamples().toArray(), 0.0001);
    }

    @Test
    public void testReadFromCSVFileSeparatesTruthCorrectly() throws IOException {
        mDataSet = new DataSet(true);
        mDataSet.addFromCSVFile(SAMPLE_FILE);
        DoubleMatrix expectedSamples = new DoubleMatrix(new double [][] {
                {1.0, 1.0},
                {1.0, 0.0},
                {0.0, 1.0},
                {0.0, 0.0}
        });

        DoubleMatrix expectedTruth = new DoubleMatrix(new double [][] {
                {1.0},
                {0.0},
                {0.0},
                {0.0}
        });

        assertTrue(mDataSet.hasTruth());
        assertEquals(1, mDataSet.getNumColsTruth());
        assertEquals(2, mDataSet.getNumColsSamples());
        assertEquals(4, mDataSet.getNumSamples());
        Assert.assertArrayEquals(expectedTruth.toArray(), mDataSet.getTruth().toArray(), 0.0001);
        Assert.assertArrayEquals(expectedSamples.toArray(), mDataSet.getSamples().toArray(), 0.0001);
    }

    @Test
    public void testGenerateRangeZeroRange() {
        DoubleMatrix initMatrix = new DoubleMatrix(new double [][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
                {10.0, 11.0, 12.0}
        });
        int start = 0;
        int end = 3;

        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
        });
        
        DoubleMatrix result = DataSet.copyRows(start, end, initMatrix);
        
        Assert.assertArrayEquals(expected.toArray(), result.toArray(), 0.001);
    }

    @Test
    public void testGenerateRangeNonZeroBasedRange() {
        DoubleMatrix initMatrix = new DoubleMatrix(new double [][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
                {10.0, 11.0, 12.0}
        });
        int start = 3;
        int end = 4;
        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {10.0, 11.0, 12.0}
        });
        
        DoubleMatrix result = DataSet.copyRows(start, end, initMatrix);
        
        Assert.assertArrayEquals(expected.toArray(), result.toArray(), 0.001);
    }
    
    @Test
    public void testSplitWorksCorrectlyWhenTruthExists() throws IOException {
        mDataSet = new DataSet(true);
        mDataSet.addFromCSVFile(SAMPLE_FILE);
        DoubleMatrix expectedTrainingSamples = new DoubleMatrix(new double [][] {
                {1.0, 1.0},
                {1.0, 0.0},
                {0.0, 1.0}
        });

        DoubleMatrix expectedTestingSamples = new DoubleMatrix(new double [][] {
                {0.0, 0.0}
        });

        DoubleMatrix expectedTrainingTruth = new DoubleMatrix(new double [][] {
                {1.0},
                {0.0},
                {0.0}
        });
        
        DoubleMatrix expectedTestingTruth = new DoubleMatrix(new double [][] {
                {0.0}
        });     
        Pair<Pair<DoubleMatrix, DoubleMatrix>, Pair<DoubleMatrix, DoubleMatrix>> result = mDataSet.split(75);
        DoubleMatrix trainingSamples = result.getLeft().getLeft();
        DoubleMatrix trainingTruth = result.getLeft().getRight();
        DoubleMatrix testingSamples = result.getRight().getLeft();
        DoubleMatrix testingTruth = result.getRight().getRight();
        
        Assert.assertArrayEquals(expectedTrainingSamples.toArray(), trainingSamples.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTrainingTruth.toArray(), trainingTruth.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTestingSamples.toArray(), testingSamples.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTestingTruth.toArray(), testingTruth.toArray(), 0.0001);
    }
    
    @Test
    public void testSplitWorksCorrectlyWhenTruthDoesNotExists() throws IOException {
        mDataSet = new DataSet(false);
        mDataSet.addFromCSVFile(SAMPLE_FILE);
        DoubleMatrix expectedTrainingSamples = new DoubleMatrix(new double [][] {
                {1.0, 1.0, 1.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
        });

        DoubleMatrix expectedTestingSamples = new DoubleMatrix(new double [][] {
                {0.0, 0.0, 0.0}
        });

        Pair<Pair<DoubleMatrix, DoubleMatrix>, Pair<DoubleMatrix, DoubleMatrix>> result = mDataSet.split(75);
        DoubleMatrix trainingSamples = result.getLeft().getLeft();
        DoubleMatrix trainingTruth = result.getLeft().getRight();
        DoubleMatrix testingSamples = result.getRight().getLeft();
        DoubleMatrix testingTruth = result.getRight().getRight();
        
        Assert.assertArrayEquals(expectedTrainingSamples.toArray(), trainingSamples.toArray(), 0.0001);
        assertTrue(trainingTruth == null);
        Assert.assertArrayEquals(expectedTestingSamples.toArray(), testingSamples.toArray(), 0.0001);
        assertTrue(testingTruth == null);
    }
}
