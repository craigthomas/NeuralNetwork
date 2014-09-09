/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.dataset;

import static org.junit.Assert.*;

import java.io.IOException;

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
}
