package ca.craigthomas.visualclassifier.dataset;

import static org.junit.Assert.*;

import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

public class TestDataSet {

    private static final String BAD_FILENAME = "/this_file_does_not_exist.csv";
    private static final String SAMPLE_FILE = "resources/small_dataset_example.csv";
    private static final DoubleMatrix SAMPLE_FILE_ALL_EXAMPLES = new DoubleMatrix(new double [][] {
            {1.0, 1.0, 1.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0}
    });
    private static final DoubleMatrix SAMPLE_FILE_EXAMPLES_ONLY = new DoubleMatrix(new double [][] {
            {1.0, 1.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {0.0, 0.0}
    });
    private static final DoubleMatrix SAMPLE_FILE_LABELS = new DoubleMatrix(new double [][] {
            {1.0},
            {0.0},
            {0.0},
            {0.0}
    });
    private DataSet mDataSet;
    
    @Test (expected=IOException.class)
    public void testReadFromNonExistentCSV() throws IOException {
        DataSet.Builder builder = new DataSet.Builder().fromCSV(BAD_FILENAME);
        builder.build();
    }
    
    @Test
    public void testReadFromCSVFileWorksCorrectly() throws IOException {
        mDataSet = new DataSet.Builder().fromCSV(SAMPLE_FILE).build();
        Assert.assertArrayEquals(SAMPLE_FILE_ALL_EXAMPLES.toArray(), mDataSet.getExamples().toArray(), 0.0001);
        assertTrue(mDataSet.getLabels() == null);
    }

    @Test
    public void testReadFromCSVFileSeparatesLabelsCorrectly() throws IOException {
        mDataSet = new DataSet.Builder().hasClassLabel().fromCSV(SAMPLE_FILE).build();
        Assert.assertArrayEquals(SAMPLE_FILE_LABELS.toArray(), mDataSet.getLabels().toArray(), 0.0001);        
        Assert.assertArrayEquals(SAMPLE_FILE_EXAMPLES_ONLY.toArray(), mDataSet.getExamples().toArray(), 0.0001);
    }
}
