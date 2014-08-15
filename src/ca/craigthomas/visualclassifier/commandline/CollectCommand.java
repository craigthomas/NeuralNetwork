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

import ca.craigthomas.visualclassifier.kinect.Monitor;
import ca.craigthomas.visualclassifier.kinect.VideoFrame;

import javax.imageio.ImageIO;

import org.kohsuke.args4j.*;
import org.openkinect.freenect.VideoFormat;

/**
 * The CollectCommand is used to collect pictures from the Kinect camera or
 * IR camera to use for training. It allows the user to specify a number of
 * command line options relating to how many pictures to take, as well as how
 * long to pause in between snapshots.
 * 
 * @author thomas
 */
public class CollectCommand extends Command {

    // The logger for the class
    private final static Logger LOGGER = Logger.getLogger(Runner.class.getName());
    // Used to generate filenames for each picture taken
    private static final String TIMESTAMP_FORMAT = "yyyy-MM-dd_HH-mm-ss";
    // Used to convert the current time into the filename string
    private DateFormat dateFormat;

    @Option(name="-i", usage="takes images using the IR camera")
    private boolean useIRCamera = false;
    
    @Option(name="-n", usage="the number of pictures to take (default 1)")
    private int numPictures = 1;
    
    @Option(name="-d", usage="delay in seconds between pictures (default 1)")
    private int delay = 1;
    
    @Option(name="-p", usage="location to store generated images")
    private String path = "./";
    
    public CollectCommand() {
         dateFormat = new SimpleDateFormat(TIMESTAMP_FORMAT);        
    }
    
    /**
     * Save the snapshot to the directory. The filename will be the current
     * date and time stamp, plus the extension of 'jpg'.
     * 
     * @param snapshot the snapshot data to save
     * @param directory the directory to save to
     */
    private void saveSnapshot(BufferedImage snapshot, File directory) {
        String filename = dateFormat.format(new Date()) + ".jpg";
        File file = new File(directory, filename);
        try {
            ImageIO.write(snapshot, "jpg", file);
        } catch (IOException e) {
            LOGGER.severe("IO exception: " + e.getMessage());
        }        
    }
    
    /**
     * Pause execution for the specified number of seconds.
     * 
     * @param seconds the number of seconds to sleep for
     */
    private void sleep(int seconds) {
        LOGGER.log(Level.INFO, "Sleeping for " + seconds + " second(s)");
        try {
            Thread.sleep(seconds * 1000);
        } catch (InterruptedException e) {
            LOGGER.warning("Sleep interruped");
        }        
    }

    @Override
    public void execute() {
        VideoFormat videoFormat = useIRCamera == true ? VideoFormat.IR_8BIT : VideoFormat.RGB;
        Monitor mFreenectMonitor = new Monitor();

        File directory = new File(path);
        if (!directory.isDirectory()) {
            LOGGER.log(Level.SEVERE, "Error: path [" + path + "] is not a directory");
            return;
        }
        
        LOGGER.log(Level.INFO, "Taking " + numPictures + " picture(s)");
        
        for (int counter = 0; counter < numPictures; counter++) {
            LOGGER.log(Level.INFO, "Taking snapshot (" + (counter+1) + " of " + numPictures + ")");
            VideoFrame videoFrame = mFreenectMonitor.takeSnapshot(videoFormat);
            saveSnapshot(videoFrame.getBufferedImage(), directory);
            sleep(delay);
        }
        
        LOGGER.log(Level.INFO, "Execution complete");
    }        
}
