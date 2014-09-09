/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.dataset;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.jblas.DoubleMatrix;

/**
 * The DataSet class is used to read data from a CSV file and optionally
 * split it into different sets (for example, test versus train). 
 * 
 * @author thomas
 */
public class DataSet {
    
    private DoubleMatrix mExamples;
    private DoubleMatrix mClasses;

    /**
     * Builds a DataSet. Must supply the data to read from, such as a 
     * CSV file.
     * 
     * @author thomas
     */
    public static class Builder {
        
        private boolean mHasClassLabel;
        private DoubleMatrix mExamples;
        private DoubleMatrix mClasses;
        
        public Builder () {
            mHasClassLabel = false;
        }
        
        /**
         * Reads the data from the specified filename - assumes that the 
         * data is in a CSV format.
         * 
         * @param filename the CSV file to read from
         * @return the builder for the DataSet
         * @throws IOException
         */
        public Builder fromCSV(String filename) throws IOException {
            readCSVFile(filename);
            return this;
        }
        
        /**
         * Indicates that the dataset in question has a class label data as
         * the last column in the dataset.
         * 
         * @return the builder for the DataSet
         */
        public Builder hasClassLabel() {
            mHasClassLabel = !mHasClassLabel;
            return this;
        }
        
        /**
         * Constructs the DataSet.
         * 
         * @return the DataSet
         */
        public DataSet build() {
            return new DataSet(this);
        }
        
        /**
         * Converts the set of inputs into a DoubleMatrix for examples, and
         * if mHasClassLabel is set, the class names.
         * 
         * @param inputs the list of inputs to convert
         */
        private void addExamples(List<List<Double>> inputs) {
            for (List<Double> inputLine : inputs) {
                double [][] values = new double[1][inputLine.size()];
                int index = 0;
                for (Double value : inputLine) {
                    values[0][index] = value.doubleValue();
                    index++;
                }
                DoubleMatrix example = new DoubleMatrix(values);
                if (mExamples == null) {
                    mExamples = example;
                } else {
                    mExamples = DoubleMatrix.concatVertically(mExamples, example);
                }
            }
        }
        
        /**
         * Converts the set of classes into a DoubleMatrix of class labels.
         * 
         * @param classes the class labels to apply
         */
        private void addClasses(List<Double> classes) {
            for (Double label : classes) {
                double [][] value = new double[1][1];
                value[0][0] = label.doubleValue();
                DoubleMatrix classLabel = new DoubleMatrix(value);
                if (mClasses == null) {
                    mClasses = classLabel;
                } else {
                    mClasses = DoubleMatrix.concatVertically(mClasses, classLabel);
                }
            }
        }

        /**
         * Generates a CSVParser for the specified filename.
         * 
         * @param filename the name of the file to parse
         * @return a CSVParser for the file
         * @throws IOException
         */
        private CSVParser createParser(String filename) throws IOException {
            File file = new File(filename);
            String fileContents = FileUtils.readFileToString(file);
            Reader reader = new StringReader(fileContents);
            return new CSVParser(reader, CSVFormat.EXCEL);            
        }

        /**
         * Reads the contents of the CSV file, and loads the data into the
         * example DoubleMatrix and class DoubleMatrix.
         * 
         * @param filename the name of the CSV file to open
         * @throws IOException
         */
        private void readCSVFile(String filename) throws IOException {
            CSVParser parser = createParser(filename);
            
            List<CSVRecord> records = parser.getRecords();
            List<List<Double>> inputs = new ArrayList<List<Double>>();
            List<Double> labels = new ArrayList<Double>();
            
            for (CSVRecord record : records) {
                List<Double> inputLine = new ArrayList<Double>();
                for (int index = 0; index < record.size(); index++) {
                    String value = record.get(index);
                    inputLine.add(Double.parseDouble(value));
                }
                if (mHasClassLabel) {
                    labels.add(inputLine.remove(record.size() - 1));
                }
                inputs.add(inputLine);
            }
            addExamples(inputs);
            if (mHasClassLabel) {
                addClasses(labels);
            }
        }
    }
    
    /**
     * Builds the DataSet object.
     * 
     * @param builder the builder for the DataSet
     */
    private DataSet(Builder builder) {
        mExamples = builder.mExamples;
        mClasses = builder.mClasses;
    }
    
    /**
     * Returns the examples.
     * 
     * @return the examples
     */
    public DoubleMatrix getExamples() {
        return mExamples;
    }
    
    /**
     * Returns the class labels.
     * 
     * @return the class labels
     */
    public DoubleMatrix getLabels() {
        return mClasses;
    }
}
