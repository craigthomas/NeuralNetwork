/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.commandline;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.jblas.DoubleMatrix;
import org.kohsuke.args4j.Option;

import ca.craigthomas.visualclassifier.dataset.DataSet;
import ca.craigthomas.visualclassifier.dataset.Prediction;
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
    
    @Option(name="-l", usage="specifies learning rate (default 0.01)")
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
    
    @Option(name="-f", usage="generate this many folds for cross-validation (default 1)")
    private int mFolds = 1;
    
    @Option(name="-l1", usage="specifies number of neurons in first hidden layer (default 10)")
    private int mLayer1 = 10;
    
    @Option(name="-l2", usage="specifies number of neurons in second hidden layer (default 0)")
    private int mLayer2 = 0;
    
    @Option(name="-o", usage="specifies number of neurons in output layer (default 1)")
    private int mOutputLayer = 1;
    
    @Option(name="--lambda", usage="specifies lambda value (default 1.0)")
    private double mLambda = 1.0;
    
    private DataSet mDataSet;
    
    public TrainCommand() {
    }
    
    /**
     * Load the data from a CSV file.
     */
    public void loadFromCSV() {
        mDataSet = new DataSet(true);
        try {
            mDataSet.addFromCSVFile(mCSVFile);
        } catch(IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage());
            mDataSet = null;
        }
    }
    
    /**
     * Load data from a directory. Assumes that all samples are images.
     * The truth value indicates whether it is a positive or negative sample.
     * 
     * @param directory the directory to load images from
     * @param truth whether the samples are positive or negative
     */
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
    
    /**
     * Loads up the files from the specified directories.
     */
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
        mDataSet.splitData(80);

        List<Integer> layerSizes = new ArrayList<Integer>();
        layerSizes.add(mDataSet.getNumColsSamples());
        if (mLayer1 != 0) {
            layerSizes.add(mLayer1);
        }
        if (mLayer2 != 0) {
            layerSizes.add(mLayer2);
        }
        layerSizes.add(mOutputLayer);
        
        Trainer trainer = new Trainer.Builder(layerSizes, mDataSet.getTrainingSet(), mDataSet.getTrainingTruth()).maxIterations(500).heartBeat(mHeartBeat).learningRate(mLearningRate).lambda(mLambda).build();
        LOGGER.log(Level.INFO, "training neural network...");
        trainer.train();
        
        NeuralNetwork model = trainer.getNeuralNetwork();
        Prediction prediction = new Prediction(model, mPredictionThreshold);
        prediction.predict(mDataSet);
        
        System.out.println("True Positives " + prediction.getTruePositives());
        System.out.println("False Positives " + prediction.getFalsePositives());
        System.out.println("True Negatives " + prediction.getTrueNegatives());
        System.out.println("False Negatives " + prediction.getFalseNegatives());
        System.out.println("Precision " + prediction.getPrecision());
        System.out.println("Recall " + prediction.getRecall());
        System.out.println("F1 " + prediction.getF1());
    }
}
