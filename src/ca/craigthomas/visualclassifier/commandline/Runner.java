/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.visualclassifier.commandline;

import java.io.IOException;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.spi.SubCommand;
import org.kohsuke.args4j.spi.SubCommandHandler;
import org.kohsuke.args4j.spi.SubCommands;

/**
 * The Runner class parses the command line, and determines what actual command
 * to run. The current commands supported are:
 * 
 *  collect - takes pictures using the camera or IR camera for training purposes
 *  
 * @author thomas
 */
public class Runner {

    // The name of the generated class
    private static final String PROGRAM_NAME = "visualclassifier.jar";

    // The set of sub-commands that the user can specify 
    @Argument(handler=SubCommandHandler.class)
    @SubCommands({
        @SubCommand(name="train", impl=TrainCommand.class),
    })
    Command command;
    
    /**
     * Parse the command line options and execute the specified command.
     * 
     * @param argv the command line arguments
     * @throws IOException
     * @throws CmdLineException
     */
    public static void main(String[] argv) throws IOException, CmdLineException {
        Runner runner = new Runner();
        CmdLineParser parser = new CmdLineParser(runner);
        try {
            parser.parseArgument(argv);
            runner.command.execute();
        } catch( CmdLineException e ) {
            System.err.println(e.getMessage());
            System.err.println("java -jar " + PROGRAM_NAME + " [options...] arguments...");
            parser.printUsage(System.err);
            System.exit(-1);
        }
    }
}
