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

/**
 * Contains static methods to read data from various sources, and return 
 * them as a list of data points.
 * 
 * @author thomas
 */
public class DataSetReader {

    /**
     * Read from a CSV file, and return the samples as a list of doubles.
     * 
     * @param filename the name of the file to read from
     * @return the list of samples from the file
     * @throws IOException
     */
    public static List<List<Double>> readCSVFile(String filename) throws IOException {
        File file = new File(filename);
        String fileContents = FileUtils.readFileToString(file);
        Reader reader = new StringReader(fileContents);
        CSVFormat format = CSVFormat.EXCEL;
        CSVParser parser = new CSVParser(reader, format);

        List<CSVRecord> records = parser.getRecords();
        List<List<Double>> inputs = new ArrayList<List<Double>>();

        for (CSVRecord record : records) {
            List<Double> inputLine = new ArrayList<Double>();
            for (int index = 0; index < record.size(); index++) {
                String value = record.get(index);
                inputLine.add(Double.parseDouble(value));
            }
            inputs.add(inputLine);
        }
        parser.close();
        return inputs;
    }
}
