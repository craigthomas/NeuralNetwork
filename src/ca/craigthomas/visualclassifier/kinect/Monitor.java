/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.kinect;

import org.openkinect.freenect.*;

/**
 * The monitor class simply connects to the Kinect, and allows the caller
 * to take a snapshot from the camera. 
 * 
 * @author thomas
 */
public class Monitor {

    private Context mContext;
    private Device mDevice;
    private VideoFrameHandler mVideoHandler;
    
    public Monitor() {
        mContext = Freenect.createContext();
        if (mContext.numDevices() == 0) {
            throw new IllegalStateException("no Kinect sensors detected");
        }
        mDevice = mContext.openDevice(0);
    }
    
    /**
     * Takes a snapshot from both the RGB video on the Kinect.
     * 
     * @throws InterruptedException
     */
    public VideoFrame takeSnapshot() throws InterruptedException {
        mDevice.setVideoFormat(VideoFormat.RGB);
        mVideoHandler = new VideoFrameHandler();
        mDevice.setLed(LedStatus.RED);
        mDevice.startVideo(mVideoHandler);
        while (mVideoHandler.getVideoFrame() == null) {
            Thread.sleep(100);
        }
        VideoFrame videoFrame = mVideoHandler.getVideoFrame();
        mDevice.stopVideo();
        mDevice.setLed(LedStatus.OFF);        
        return videoFrame;
    }
}
