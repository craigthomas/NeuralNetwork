/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.dataset;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Assert;
import org.junit.Test;

public class TestDataSetReader {

    private static final String BAD_FILENAME = "/this_file_does_not_exist.csv";
    private static final String SAMPLE_FILE = "resources/small_dataset_example.csv";
    
    @Test (expected=IOException.class)
    public void testReadFromNonExistentCSV() throws IOException {
        DataSetReader.readCSVFile(BAD_FILENAME);
    }
    
    @Test
    public void testReadFromCSVFileSeparatesLabelsCorrectly() throws IOException {
        double [][] expected = {
                {1.0, 1.0, 1.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0}                
        };
        
        List<List<Double>> result = DataSetReader.readCSVFile(SAMPLE_FILE);
        
        assertEquals(expected.length, result.size());
        
        for (int index = 0; index < expected.length; index++) {
            List<Double> row = result.get(index);
            Double [] temp = row.toArray(new Double[row.size()]);
            double [] sampleRow = ArrayUtils.toPrimitive(temp);
            Assert.assertArrayEquals(expected[index], sampleRow, 0.0001);        
        }
    }
}
