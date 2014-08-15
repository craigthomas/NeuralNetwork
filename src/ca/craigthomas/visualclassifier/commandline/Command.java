/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.commandline;

/**
 * Commands represent some form of action that the java program can execute.
 * As a minimum, each command must have an <code>execute</code> function that
 * is responsible for actually performing the command specified.
 * 
 * @author thomas
 */
public abstract class Command {
    
    /**
     * Runs the specified command.
     */
    public abstract void execute();
}
