/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.dataset;

import java.io.IOException;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.jblas.DoubleMatrix;

/**
 * The DataSet class is used to read data from various sources. The DataSet
 * class keeps track of two types of data: the actual example inputs called
 * Samples, and their optional output labels called Truth. 
 * 
 * @author thomas
 */
public class DataSet {
    
    private DoubleMatrix mSamples;
    private DoubleMatrix mTruth;
    private final boolean sHasTruth;
    
    /**
     * The DataSet constructor. If hasTruth is set to true, then the
     * class will expect ground truth to be passed in with the data.
     * 
     * @param hasTruth whether the Samples have ground truth information
     */
    public DataSet(boolean hasTruth) {
        sHasTruth = hasTruth;
    }
    
    /**
     * Returns the Samples.
     * 
     * @return the Samples
     */
    public DoubleMatrix getSamples() {
        return mSamples;
    }
    
    /**
     * Returns the ground truth.
     * 
     * @return the ground truth
     */
    public DoubleMatrix getTruth() {
        return mTruth;
    }
    
    /**
     * Get the number of columns in the Samples.
     * 
     * @return the number of columns in the Samples
     */
    public int getNumColsSamples() {
        if (mSamples == null) {
            return 0;
        }
        return mSamples.columns;
    }
    
    /**
     * Returns the number of columns in the Truth data.
     * 
     * @return return the number of columns in the Truth data
     */
    public int getNumColsTruth() {
        if (mTruth == null) {
            return 0;
        }
        return mTruth.columns;
    }
    
    /**
     * Returns the number of samples (rows) in the DataSet.
     * 
     * @return the number of samples in the DataSet
     */
    public int getNumSamples() {
        if (mSamples == null) {
            return 0;
        }
        return mSamples.rows;
    }
    
    /**
     * Returns true if the DataSet has ground truth associated with it,
     * false otherwise.
     * 
     * @return true if the DataSet has ground truth
     */
    public boolean hasTruth() {
        return sHasTruth;
    }
    
    /**
     * Reads samples from a CSV file, and adds them to the DataSet. If header
     * is set (true), will ignore the first line of the file.
     * 
     * @param filename the name of the file to read from
     * @param hasHeader whether the file has a header line
     * @throws IOException
     */
    public void addFromCSVFile(String filename) throws IOException {
        List<List<Double>> data = DataSetReader.readCSVFile(filename);
        addSamples(data);
    }
    
    /**
     * Adds a list of samples to the DataSet. Each element in the list contains
     * a list of Doubles, which are assumed to be the samples to add. If the
     * DataSet has ground truth, the last element in each of the list of samples
     * is assumed to be the ground truth label.
     * 
     * @param samples the list of samples to add
     */
    public void addSamples(List<List<Double>> samples) {
        for (List<Double> row : samples) {
            Double [] temp = row.toArray(new Double[row.size()]);
            double [] sampleRow = ArrayUtils.toPrimitive(temp);
            if (sHasTruth) {
                addSampleRow(new DoubleMatrix(
                        ArrayUtils.subarray(sampleRow, 0, sampleRow.length-1)));
                addTruthRow(new DoubleMatrix(
                        ArrayUtils.subarray(sampleRow, sampleRow.length-1, sampleRow.length)));
            } else {
                addSampleRow(new DoubleMatrix(sampleRow));
            }
        } 
    }

    /**
     * Internal helper function that will add a row of samples to the DataSet.
     * 
     * @param samples the matrix of samples to add
     */
    private void addSampleRow(DoubleMatrix samples) {
        mSamples = (mSamples == null) ? samples : DoubleMatrix.concatVertically(mSamples, samples);
    }
    
    /**
     * Internal helper function that will add a row of ground truth labels
     * to the DataSet.
     * 
     * @param truth the matrix of truth values to add
     */
    private void addTruthRow(DoubleMatrix truth) {
        mTruth = (mTruth == null) ? truth : DoubleMatrix.concatVertically(mTruth, truth);
    }
}
