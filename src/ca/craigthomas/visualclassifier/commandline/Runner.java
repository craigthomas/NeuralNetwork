/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.commandline;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.imageio.ImageIO;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import ca.craigthomas.visualclassifier.kinect.Monitor;
import ca.craigthomas.visualclassifier.kinect.VideoFrame;

public class Runner {

    private static final String DELAY_OPTION = "d";
    private static final String NUMBER_OPTION = "n";
    private static final String HELP_OPTION = "h";
    private static final String PATH_OPTION = "p";
    private static final String IR_OPTION = "i";
    private static final String PROGRAM_NAME = "visualclassifier";
    private final static Logger LOGGER = Logger.getLogger(Runner.class.getName());
    private static final String TIMESTAMP_CONVERSION = "yyyy-MM-dd_HH-mm-ss";

    /**
     * Generates the set of options for the command line option parser.
     * 
     * @return The options for the emulator
     */
    public static Options generateOptions() {
        Options options = new Options();
        
        @SuppressWarnings("static-access")
        Option ir = OptionBuilder
                .withDescription("takes images using the IR camera")
                .create(IR_OPTION);

        @SuppressWarnings("static-access")
        Option delay = OptionBuilder
                .withArgName("delay")
                .hasArg()
                .withDescription("sets the delay in seconds between pictures (default 1)")
                .create(DELAY_OPTION);

        @SuppressWarnings("static-access")
        Option number = OptionBuilder
                .withArgName("number")
                .hasArg()
                .withDescription("the number of pictures to take (default 1)")
                .create(NUMBER_OPTION);
        
        @SuppressWarnings("static-access")
        Option path = OptionBuilder
                .withArgName("path")
                .hasArg()
                .withDescription("location to store generated images in")
                .create(PATH_OPTION);
        

        @SuppressWarnings("static-access")
        Option help = OptionBuilder.withDescription(
                "show this help message and exit").create(HELP_OPTION);

        options.addOption(help);
        options.addOption(delay);
        options.addOption(path);
        options.addOption(number);
        options.addOption(ir);
        return options;
    }

    /**
     * Attempts to parse the command line options.
     * 
     * @param args
     *            The set of arguments provided to the program
     * @return A CommandLine object containing the parsed options
     */
    public static CommandLine parseCommandLineOptions(String[] args) {
        CommandLineParser parser = new BasicParser();
        try {
            return parser.parse(generateOptions(), args);
        } catch (ParseException e) {
            System.err.println("Error: Command line parsing failed.");
            System.err.println("Reason: " + e.getMessage());
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(PROGRAM_NAME, generateOptions());
            System.exit(1);
        }
        return null;
    }
    
    public static void main(String[] argv) throws InterruptedException, IOException {
        VideoFrame videoFrame;
        boolean irCamera = false;
        DateFormat dateFormat = new SimpleDateFormat(TIMESTAMP_CONVERSION);
        int numPictures = 1;
        int delay = 1;
        String path = "./";
        CommandLine commandLine = parseCommandLineOptions(argv);
        
        if (commandLine.hasOption(HELP_OPTION)) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(PROGRAM_NAME, generateOptions());
            return;
        }

        if (commandLine.hasOption(NUMBER_OPTION)) {
            numPictures = Integer.parseInt(commandLine.getOptionValue(NUMBER_OPTION));
        }
        
        if (commandLine.hasOption(DELAY_OPTION)) {
            delay = Integer.parseInt(commandLine.getOptionValue(DELAY_OPTION));
        }
        
        if (commandLine.hasOption(IR_OPTION)) {
            irCamera = true;
        }

        if (commandLine.hasOption(PATH_OPTION)) {
            path = commandLine.getOptionValue(PATH_OPTION);
        }
        
        // Make sure the path specified is valid
        File directory = new File(path);
        if (!directory.isDirectory()) {
            LOGGER.log(Level.SEVERE, "Error: path [" + path + "] is not a directory");
            return;
        }
        
        delay = delay * 1000;        
        Monitor mFreenectMonitor = new Monitor();
        LOGGER.log(Level.INFO, "Taking " + numPictures + " picture(s)");
        
        for (int counter = 0; counter < numPictures; counter++) {
            LOGGER.log(Level.INFO, "Taking snapshot (" + (counter+1) + " of " + numPictures + ")");
            if (irCamera) {
                videoFrame = mFreenectMonitor.takeIRSnapshot();
            } else {
                videoFrame = mFreenectMonitor.takeSnapshot();
            }
            BufferedImage snapshot = videoFrame.getBufferedImage();
            String filename = dateFormat.format(new Date()) + ".jpg";
            File file = new File(directory, filename);
            ImageIO.write(snapshot, "jpg", file);
            LOGGER.log(Level.INFO, "Sleeping for " + delay/1000 + " second(s)");
            Thread.sleep(delay);
        }
        
        LOGGER.log(Level.INFO, "Execution complete");
    }
}
