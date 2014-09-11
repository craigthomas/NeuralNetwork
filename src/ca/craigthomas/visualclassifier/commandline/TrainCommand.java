/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.commandline;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;
import org.kohsuke.args4j.Option;

import ca.craigthomas.visualclassifier.dataset.DataSet;
import ca.craigthomas.visualclassifier.imageprocessing.Image;
import ca.craigthomas.visualclassifier.nn.network.NeuralNetwork;
import ca.craigthomas.visualclassifier.nn.trainer.Trainer;

/**
 * The TrainCommand is used to train a neural network based upon a number of
 * positive and negative examples. 
 * 
 * @author thomas
 */
public class TrainCommand extends Command {

    // The logger for the class
    private final static Logger LOGGER = Logger.getLogger(Runner.class.getName());
    // Used to generate filenames for each picture taken

    @Option(name="-b", usage="specifies heartbeat during training (default 100 iterations)")
    private int mHeartBeat = 100;
    
    @Option(name="-a", usage="specifies learning rate (default 0.01)")
    private double mLearningRate = 0.01;
    
    @Option(name="-c", usage="loads data from a CSV file")
    private String mCSVFile = "";
    
    @Option(name="-p", usage="specifies positive image directory")
    private String mPositiveDir = "";
    
    @Option(name="-n", usage="specifies negative image directory")
    private String mNegativeDir = "";
    
    @Option(name="-w", usage="ensure images have specified width in pixels (default 10)")
    private int mRequiredWidth = 10;
    
    @Option(name="-h", usage="ensure images have specified height in pixels (default 10)")
    private int mRequiredHeight = 10;
    
    @Option(name="-s", usage="save prediction results into specified directory")
    private String mSaveDir = "";
    
    @Option(name="-t", usage="prediction threshold (default 0.5)")
    private double mPredictionThreshold = 0.5;
    
    private DataSet mDataSet;
    
    public TrainCommand() {
    }
    
    public void loadFromCSV() {
        mDataSet = new DataSet(true);
        try {
            mDataSet.addFromCSVFile(mCSVFile);
        } catch(IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage());
            mDataSet = null;
        }
    }
    
    public void loadFromDirectory(File directory, double truth) {
        File [] files = directory.listFiles();
        for (File file : files) {
            String filename = file.getAbsolutePath();
            Image image = new Image(filename);
            
            if (image.getWidth() != mRequiredWidth || image.getHeight() != mRequiredHeight) {
                LOGGER.log(Level.WARNING, "file " + filename + " not correct size, skipping (want " + mRequiredWidth + "x" + mRequiredHeight + ", got " + image.getWidth() + "x" + image.getHeight() + ")");
            } else {
                mDataSet.addSample(image.convertGrayscaleToMatrix(truth));
            }
        }
    }
    
    public void loadFromDirectories() {
        File positiveDir = new File(mPositiveDir);
        File negativeDir = new File(mNegativeDir);
        
        if (!positiveDir.isDirectory()) {
            LOGGER.log(Level.SEVERE, "positives directory [" + mPositiveDir + "] is not a directory");
            return;
        }
    
        if (!negativeDir.isDirectory()) {
            LOGGER.log(Level.SEVERE, "negatives directory [" + mNegativeDir + "] is not a directory");
            return;
        }
        
        mDataSet = new DataSet(true);
        loadFromDirectory(positiveDir, 1.0);
        loadFromDirectory(negativeDir, 0.0);
    }
    
    public void saveResults(DoubleMatrix actual, DoubleMatrix predictions, DoubleMatrix samples) {
        File directory = new File(mSaveDir);
        if (!directory.isDirectory()) {
            LOGGER.log(Level.SEVERE, "save directory [" + mSaveDir + "] is not a directory");
            return;
        }
   }
    
    @Override
    public void execute() {
        if (!mCSVFile.isEmpty()) {
            loadFromCSV();
        } else {
            loadFromDirectories();
        }
        
        if (mDataSet == null) {
            LOGGER.log(Level.SEVERE, "no data set could be built, exiting");
            return;
        }
        
        LOGGER.log(Level.INFO, "loaded " + mDataSet.getNumSamples() + " sample(s)");
        mDataSet.randomize();
        Pair<Pair<DoubleMatrix, DoubleMatrix>, Pair<DoubleMatrix, DoubleMatrix>> trainingTestSplit = mDataSet.split(80);
        DoubleMatrix trainingSamples = trainingTestSplit.getLeft().getLeft();
        DoubleMatrix trainingTruth = trainingTestSplit.getLeft().getRight();
        DoubleMatrix testingSamples = trainingTestSplit.getRight().getLeft();
        DoubleMatrix testingTruth = trainingTestSplit.getRight().getRight();
        List<Integer> layerSizes = Arrays.asList(3600, 10, 1);
        
        Trainer trainer = new Trainer.Builder(layerSizes, trainingSamples, trainingTruth).maxIterations(500).build();
        LOGGER.log(Level.INFO, "training neural network...");
        trainer.train();
        
        NeuralNetwork model = trainer.getNeuralNetwork();
        DoubleMatrix predictions = model.predict(testingSamples);
        
        double truePositives = 0.0;
        double falsePositives = 0.0;
        double trueNegatives = 0.0;
        double falseNegatives = 0.0;
        
        for (int index = 0; index < predictions.rows; index++) {
            int prediction = (predictions.get(index, 0) > mPredictionThreshold) ? 1 : 0;
            int actual = (testingTruth.get(index, 0) > mPredictionThreshold) ? 1 : 0;
            if (actual == 1) {
                if (prediction == actual) {
                    truePositives += 1.0;
                } else {
                    falseNegatives += 1.0;
                }
            } else {
                if (prediction == actual) {
                    trueNegatives += 1.0;
                } else {
                    falsePositives += 1.0;
                }
            }
        }
        
        System.out.println("True Positives " + truePositives);
        System.out.println("False Positives " + falsePositives);
        System.out.println("True Negatives " + trueNegatives);
        System.out.println("False Negatives " + falseNegatives);
        
        double precision = truePositives / (truePositives + falsePositives);
        double recall = truePositives / (truePositives + falseNegatives);
        
        System.out.println("Precision " + precision);
        System.out.println("Recall " + recall);
    }
}
