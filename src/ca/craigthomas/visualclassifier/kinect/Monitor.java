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
     * Takes a snapshot from the IR or RGB camera. The type taken depends on
     * the format specified in <code>videoFormat</code>. Returns a single
     * VideoFrame with the snapshot information contained within it.
     * 
     * @throws InterruptedException
     * @return a single frame photo
     */
    public VideoFrame takeSnapshot(VideoFormat videoFormat) {
        mDevice.setVideoFormat(videoFormat);
        mVideoHandler = new VideoFrameHandler();
        mDevice.setLed(LedStatus.RED);
        mDevice.startVideo(mVideoHandler);
        while (mVideoHandler.getVideoFrame() == null) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                
            }
        }
        VideoFrame videoFrame = mVideoHandler.getVideoFrame();
        mDevice.stopVideo();
        mDevice.setLed(LedStatus.OFF);        
        return videoFrame;
    }
}
