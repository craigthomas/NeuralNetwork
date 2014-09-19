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

import javax.imageio.ImageIO;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.stat.StatUtils;
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

    @Option(name="-b", usage="specifies heartbeat during training (default 100 iterations)")
    private int mHeartBeat = 100;
    
    @Option(name="-l", usage="specifies learning rate (default 0.01)")
    private double mLearningRate = 0.01;
    
    @Option(name="-c", usage="loads data from a CSV file")
    private String mCSVFile = "";
    
    @Option(name="--color", usage="processes images in color")
    private boolean mColor = false;
    
    @Option(name="-p", usage="specifies positive image directory")
    private String mPositiveDir = "";
    
    @Option(name="-n", usage="specifies negative image directory")
    private String mNegativeDir = "";
    
    @Option(name="-w", usage="ensure images have specified width in pixels (default 10)")
    private int mRequiredWidth = 10;
    
    @Option(name="-h", usage="ensure images have specified height in pixels (default 10)")
    private int mRequiredHeight = 10;
    
    @Option(name="--save", usage="save prediction results into specified directory")
    private String mSaveDir = "";
    
    @Option(name="-s", usage="splits the data between training and testing (default 80 training)")
    private int mSplit = 80;
    
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
    
    @Option(name="-i", usage="number of iterations (default 500)")
    private int mIterations = 500;
     
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
            LOGGER.log(Level.INFO, "loaded " + mDataSet.getNumSamples() + " sample(s)");
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
                if (mColor) {
                    mDataSet.addSample(image.convertColorToMatrix(truth));
                } else {
                    mDataSet.addSample(image.convertGrayscaleToMatrix(truth));                    
                }
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
        LOGGER.log(Level.INFO, "loaded " + mDataSet.getNumSamples() + " sample(s)");
    }
    
    public void saveImage(Image image, File path, String filename) {
        File saveFile = new File(path, filename);
        try {
            ImageIO.write(image.getBufferedImage(), "png", saveFile);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "could not save file [" + saveFile.getAbsolutePath() + "]");
        }
    }
    
    public void saveResults(NeuralNetwork bestModel, DataSet bestFold) {
        File directory = new File(mSaveDir);
        if (!directory.isDirectory()) {
            LOGGER.log(Level.SEVERE, "save directory [" + mSaveDir + "] is not a directory");
            return;
        }
        
        Prediction predictions = new Prediction(bestModel, mPredictionThreshold);
        predictions.predict(bestFold);
        DoubleMatrix falsePositives = predictions.getFalsePositiveSamples();
        DoubleMatrix falseNegatives = predictions.getFalseNegativeSamples();
        for (int i = 0; i < falsePositives.rows; i++) {
            Image image = new Image(falsePositives.getRow(i), mRequiredWidth, mRequiredHeight, mColor);
            saveImage(image, directory, "fp" + (i+1) + ".png");
        }
        for (int i = 0; i < falseNegatives.rows; i++) {
            Image image = new Image(falseNegatives.getRow(i), mRequiredWidth, mRequiredHeight, mColor);
            saveImage(image, directory, "fn" + (i+1) + ".png");
        }
    }
    
    @Override
    public void execute() {
        NeuralNetwork bestModel = null;
        DataSet bestFold = null;
        double [] tp = new double [mFolds];
        double [] fp = new double [mFolds];
        double [] tn = new double [mFolds];
        double [] fn = new double [mFolds];
        double [] precision = new double [mFolds];
        double [] recall = new double [mFolds];
        double [] f1 = new double [mFolds];
        double bestF1 = 0;
        
        // Step 1: create the dataset
        if (!mCSVFile.isEmpty()) {
            loadFromCSV();
        } else {
            loadFromDirectories();
        }
        
        if (mDataSet == null) {
            LOGGER.log(Level.SEVERE, "no data set could be built, exiting");
            return;
        }
        
        // Step 2: Generate layer information
        List<Integer> layerSizes = new ArrayList<Integer>();
        layerSizes.add(mDataSet.getNumColsSamples());
        if (mLayer1 != 0) {
            layerSizes.add(mLayer1);
        }
        if (mLayer2 != 0) {
            layerSizes.add(mLayer2);
        }
        layerSizes.add(mOutputLayer);
        
        // Step 3: generate the folds and train the model
        for (int fold = 0; fold < mFolds; fold++) {
            LOGGER.log(Level.INFO, "processing fold " + (fold+1));
            LOGGER.log(Level.INFO, "randomizing dataset");
            mDataSet.randomize();
            LOGGER.log(Level.INFO, "generating training and testing sets");
            Pair<DataSet, DataSet> split = mDataSet.splitSequentially(mSplit);
            DataSet trainingData = split.getLeft();
            DataSet testingData = split.getRight();
            LOGGER.log(Level.INFO, "training neural network...");            
            Trainer trainer = new Trainer.Builder(layerSizes, trainingData).maxIterations(mIterations).heartBeat(mHeartBeat).learningRate(mLearningRate).lambda(mLambda).build();
            trainer.train();
            
            // Step 4: evaluate each model
            NeuralNetwork model = trainer.getNeuralNetwork();
            Prediction prediction = new Prediction(model, mPredictionThreshold);
            prediction.predict(testingData);
            System.out.println("True Positives " + prediction.getTruePositives());
            System.out.println("False Positives " + prediction.getFalsePositives());
            System.out.println("True Negatives " + prediction.getTrueNegatives());
            System.out.println("False Negatives " + prediction.getFalseNegatives());
            System.out.println("Precision " + prediction.getPrecision());
            System.out.println("Recall " + prediction.getRecall());
            System.out.println("F1 " + prediction.getF1());
            
            tp[fold] = prediction.getTruePositives();
            fp[fold] = prediction.getFalsePositives();
            tn[fold] = prediction.getTrueNegatives();
            fn[fold] = prediction.getFalseNegatives();
            precision[fold] = prediction.getPrecision();
            recall[fold] = prediction.getRecall();
            f1[fold] = prediction.getF1();
            if (f1[fold] > bestF1) {
                bestModel = model;
                bestFold = mDataSet.dup();
                bestF1 = f1[fold];
            }
        }
        
        // Step 6: save the best information to the specified directory
        if (!mSaveDir.isEmpty()) {
            saveResults(bestModel, bestFold);
        }
        
        // Step 5: compute the overall statistics
        System.out.println("Overall Statistics");
        System.out.println("True Positives " + StatUtils.mean(tp) + " (" + StatUtils.variance(tp) + ")");
        System.out.println("False Positives " + StatUtils.mean(fp) + " (" + StatUtils.variance(fp) + ")");
        System.out.println("True Negatives " + StatUtils.mean(tn) + " (" + StatUtils.variance(tn) + ")");
        System.out.println("False Negatives " + StatUtils.mean(fn) + " (" + StatUtils.variance(fn) + ")");
        System.out.println("Precision " + StatUtils.mean(precision) + " (" + StatUtils.variance(precision) + ")");
        System.out.println("Recall " + StatUtils.mean(recall) + " (" + StatUtils.variance(recall) + ")");
        System.out.println("F1 " + StatUtils.mean(f1) + " (" + StatUtils.variance(f1) + ")");
    }
}
