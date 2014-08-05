/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.kinect;

import java.nio.ByteBuffer;

import org.openkinect.freenect.*;

/**
 * A simple class to store the last frame of video captured.
 * 
 * @author thomas
 */
public class VideoFrameHandler implements VideoHandler {
    
    private VideoFrame mVideoFrame;
    
    public VideoFrameHandler() {}
    
    @Override
    public void onFrameReceived(FrameMode arg0, ByteBuffer arg1, int arg2) {
        mVideoFrame = new VideoFrame(arg0, arg1, arg2);
    }
    
    /**
     * Returns the last captured video frame.
     * 
     * @return the last captured video frame
     */
    public VideoFrame getVideoFrame() {
        return mVideoFrame;
    }
}
