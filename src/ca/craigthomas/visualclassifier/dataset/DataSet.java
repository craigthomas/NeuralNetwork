/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.dataset;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
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
    private Random mRandom;
    private DoubleMatrix mTestingSamples;
    private DoubleMatrix mTrainingSamples;
    private DoubleMatrix mTestingTruth;
    private DoubleMatrix mTrainingTruth;
    
    /**
     * The DataSet constructor. If hasTruth is set to true, then the
     * class will expect ground truth to be passed in with the data.
     * 
     * @param hasTruth whether the Samples have ground truth information
     */
    public DataSet(boolean hasTruth) {
        sHasTruth = hasTruth;
        mRandom = new Random();
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
            double [][] matrix = new double[1][sampleRow.length];
            if (sHasTruth) {
                matrix[0] = ArrayUtils.subarray(sampleRow, 0, sampleRow.length-1);
                addSampleRow(new DoubleMatrix(matrix));
                matrix[0] = ArrayUtils.subarray(sampleRow, sampleRow.length-1, sampleRow.length);
                addTruthRow(new DoubleMatrix(matrix));
            } else {
                matrix[0] = sampleRow;
                addSampleRow(new DoubleMatrix(matrix));
            }
        } 
    }
    
    /**
     * Adds a single row DoubleMatrix set of values to the list of samples.
     *  
     * @param sample the DoubleMatrix column vector to add
     */
    public void addSample(DoubleMatrix sample) {
        List<List<Double>> newSampleList = new ArrayList<List<Double>>();
        double [] values = new double [sample.columns];
        for (int index = 0; index < sample.columns; index++) {
            values[index] = sample.get(0, index);
        }
        newSampleList.add(Arrays.asList(ArrayUtils.toObject(values)));
        addSamples(newSampleList);
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
    
    /**
     * Randomizes the data points within the DataSet.
     */
    public void randomize() {
        for (int counter = 0; counter < mSamples.rows * 5; counter++) {
            int firstIndex = mRandom.nextInt(mSamples.rows);
            int secondIndex = mRandom.nextInt(mSamples.rows);
            DoubleMatrix tempRow = mSamples.getRow(firstIndex);
            DoubleMatrix tempRow1 = mSamples.getRow(secondIndex);
            mSamples.putRow(firstIndex, tempRow1);
            mSamples.putRow(secondIndex, tempRow);
            
            if (sHasTruth) {
                tempRow = mTruth.getRow(firstIndex);
                tempRow1 = mTruth.getRow(secondIndex);
                mTruth.putRow(firstIndex, tempRow1);
                mTruth.putRow(secondIndex, tempRow);                
            }
        }
    }
    
    /**
     * A workaround for the JBlas library IntervalRange class - specifying 
     * ranges that don't start at zero have a problem. Copy the given rows
     * instead.
     * 
     * @param a the start of the range (inclusive)
     * @param b the end of the range (exclusive)
     * @param matrix the matrix to copy from
     * @return a new matrix with the rows from a to b
     */
    protected static DoubleMatrix copyRows(int a, int b, DoubleMatrix matrix) {
        DoubleMatrix result = null;
        for (int index = a; index < b; index++) {
            if (result == null) {
                result = matrix.getRow(index).dup();
            } else {
                result = DoubleMatrix.concatVertically(result, matrix.getRow(index).dup());
            }
        }
        return result;
    }
    
    /**
     * Splits a DataSet into two sets - a training and a testing set - based
     * upon the percentage. For example, a percentage of 60 would allocate 
     * 60% to the training set and 40% to the testing set. 
     * 
     * @param percentage the percentage to put into the training set
     */
    public void splitData(int percentage) {
        Pair<Pair<DoubleMatrix, DoubleMatrix>, Pair<DoubleMatrix, DoubleMatrix>> split = split(percentage);
        mTrainingSamples = split.getLeft().getLeft();
        mTestingSamples = split.getRight().getLeft();
        mTrainingTruth = split.getLeft().getRight();
        mTestingTruth = split.getRight().getRight();
    }
    
    public DoubleMatrix getTrainingSet() {
        return mTrainingSamples;
    }
    
    public DoubleMatrix getTrainingTruth() {
        return mTrainingTruth;
    }
    
    public DoubleMatrix getTestingSet() {
        return mTestingSamples;
    }
    
    public DoubleMatrix getTestingTruth() {
        return mTestingTruth;
    }
    
    /**
     * Splits a DataSet into two sets - a training and a testing set - based
     * upon the percentage. For example, a percentage of 60 would allocate 
     * 60% to the training set and 40% to the testing set. Returns a pair of 
     * pairs - the first pair is the training set, the second pair is the
     * testing set. Within the training and testing pairs, the first DoubleMatrix
     * is the sample data, while the second DoubleMatrix are the ground truth
     * values.
     * 
     * @param percentage the percentage to put into the training set
     * @return a pair of pairs, where the left is training, right is testing
     */
    protected Pair<Pair<DoubleMatrix, DoubleMatrix>, Pair<DoubleMatrix, DoubleMatrix>> split(int percentage) {
        int trainStart = 0; 
        int trainEnd = (int)((percentage / 100.0) * (float)mSamples.rows);
        int testStart = trainEnd;
        int testEnd = mSamples.rows;
        DoubleMatrix trainingSamples = copyRows(trainStart, trainEnd, mSamples);
        DoubleMatrix testingSamples = copyRows(testStart, testEnd, mSamples);
        DoubleMatrix trainingTruth = null;
        DoubleMatrix testingTruth = null;
        if (sHasTruth) {
            trainingTruth = copyRows(trainStart, trainEnd, mTruth);
            testingTruth = copyRows(testStart, testEnd, mTruth);
        }
        Pair<DoubleMatrix, DoubleMatrix> trainingPair = Pair.of(trainingSamples, trainingTruth);
        Pair<DoubleMatrix, DoubleMatrix> testingPair = Pair.of(testingSamples, testingTruth);
        return Pair.of(trainingPair, testingPair);
    }
}
