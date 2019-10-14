/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.commandline;

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

import ca.craigthomas.neuralnetwork.dataset.DataSet;
import ca.craigthomas.neuralnetwork.dataset.Prediction;
import ca.craigthomas.neuralnetwork.imageprocessing.Image;
import ca.craigthomas.neuralnetwork.components.network.NeuralNetwork;
import ca.craigthomas.neuralnetwork.components.trainer.Trainer;

/**
 * The TrainCommand is used to train a neural network based upon a number of
 * positive and negative examples. 
 */
public class TrainCommand
{
    // The logger for the class
    private final static Logger LOGGER = Logger.getLogger(Runner.class.getName());
    // The underlying data set
    private DataSet mDataSet;
    // The arguments passed to the command
    TrainArguments arguments;
    
    public TrainCommand(TrainArguments arguments) {
        this.arguments = arguments;
    }
    
    /**
     * Load the data from a CSV file.
     */
    public void loadFromCSV() {
        mDataSet = new DataSet(true);
        try {
            mDataSet.addFromCSVFile(arguments.csvFile);
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
            
            if (image.getWidth() != arguments.requiredWidth || image.getHeight() != arguments.requiredHeight) {
                LOGGER.log(Level.WARNING, "file " + filename + " not correct size, skipping (want " + arguments.requiredWidth + "x" + arguments.requiredHeight + ", got " + image.getWidth() + "x" + image.getHeight() + ")");
            } else {
                if (arguments.color) {
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
        File positiveDir = new File(arguments.positiveDir);
        File negativeDir = new File(arguments.negativeDir);
        
        if (!positiveDir.isDirectory()) {
            LOGGER.log(Level.SEVERE, "positives directory [" + arguments.positiveDir + "] is not a directory");
            return;
        }
    
        if (!negativeDir.isDirectory()) {
            LOGGER.log(Level.SEVERE, "negatives directory [" + arguments.negativeDir + "] is not a directory");
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
        File directory = new File(arguments.saveDir);
        if (!directory.isDirectory()) {
            LOGGER.log(Level.SEVERE, "save directory [" + arguments.saveDir + "] is not a directory");
            return;
        }
        
        Prediction predictions = new Prediction(bestModel, arguments.predictionThreshold);
        predictions.predict(bestFold);
        DoubleMatrix falsePositives = predictions.getFalsePositiveSamples();
        DoubleMatrix falseNegatives = predictions.getFalseNegativeSamples();
        for (int i = 0; i < falsePositives.rows; i++) {
            Image image = new Image(falsePositives.getRow(i), arguments.requiredWidth, arguments.requiredHeight, arguments.color);
            saveImage(image, directory, "fp" + (i+1) + ".png");
        }
        for (int i = 0; i < falseNegatives.rows; i++) {
            Image image = new Image(falseNegatives.getRow(i), arguments.requiredWidth, arguments.requiredHeight, arguments.color);
            saveImage(image, directory, "fn" + (i+1) + ".png");
        }
    }
    
    public void execute() {
        NeuralNetwork bestModel = null;
        DataSet bestFold = null;
        double [] tp = new double [arguments.folds];
        double [] fp = new double [arguments.folds];
        double [] tn = new double [arguments.folds];
        double [] fn = new double [arguments.folds];
        double [] precision = new double [arguments.folds];
        double [] recall = new double [arguments.folds];
        double [] f1 = new double [arguments.folds];
        double bestF1 = 0;
        
        // Step 1: create the dataset
        if (!arguments.csvFile.isEmpty()) {
            loadFromCSV();
        } else {
            loadFromDirectories();
        }
        
        if (mDataSet == null) {
            LOGGER.log(Level.SEVERE, "no data set could be built, exiting");
            return;
        }
        
        // Step 2: Generate layer information
        List<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(mDataSet.getNumColsSamples());
        if (arguments.layer1 != 0) {
            layerSizes.add(arguments.layer1);
        }
        if (arguments.layer2 != 0) {
            layerSizes.add(arguments.layer2);
        }
        layerSizes.add(arguments.outputLayer);
        
        // Step 3: generate the folds and train the model
        for (int fold = 0; fold < arguments.folds; fold++) {
            LOGGER.log(Level.INFO, "processing fold " + (fold+1));
            LOGGER.log(Level.INFO, "randomizing dataset");
            mDataSet.randomize();
            LOGGER.log(Level.INFO, "generating training and testing sets");
            Pair<DataSet, DataSet> split = mDataSet.splitEqually(arguments.split);
            DataSet trainingData = split.getLeft();
            DataSet testingData = split.getRight();
            LOGGER.log(Level.INFO, "training neural network...");   
            trainingData.randomize();
            Trainer trainer = new Trainer.Builder(layerSizes, trainingData)
                    .maxIterations(arguments.iterations)
                    .heartBeat(arguments.heartBeat)
                    .learningRate(arguments.learningRate)
                    .lambda(arguments.lambda).build();
            trainer.train();
            
            // Step 4: evaluate each model
            NeuralNetwork model = trainer.getNeuralNetwork();
            Prediction prediction = new Prediction(model, arguments.predictionThreshold);
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
        if (!arguments.saveDir.isEmpty()) {
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
