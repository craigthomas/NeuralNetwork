# ML Deer Detector

[![Build Status](https://travis-ci.org/craigthomas/DeerDetector.svg?branch=master)](https://travis-ci.org/craigthomas/MLDeerDetector)[![Coverage Status](https://coveralls.io/repos/craigthomas/DeerDetector/badge.png)](https://coveralls.io/r/craigthomas/DeerDetector)

## What is it?

This project represents a machine learning algorithm and support programs
to detect the presence of deer within a picture. The project contains code
to connect to an XBOX 360 Kinect and take pictures. The pictures are then
passed through a trained machine learning algorithm that classifies whether
a deer exists in the picture or not.

The project contains code to:

* Take a set number of pictures with the Kinect camera
* Train a machine learning model to detect deer within a set of pictures
* Use the Kinect to determine if a picture contains a deer

For more information on the project, [read my blog posts on the
project](#)

## License

This project makes use of an MIT style license. Please see the file called 
LICENSE for more information. Note that this project may make use of other
software that has separate license terms. See the section called `Third
Party Licenses and Attributions` below for more information on those
software components.


## Compiling

Simply copy the source files to a directory of your choice. In addition
to the source, you will need the following required software packages:

* [Java JDK 7](http://www.oracle.com/technetwork/java/javase/downloads/index.html) 1.7.0 u51 or later
* [OpenKinect libfreenect](http://openkinect.org/wiki/Getting_Started)

To build the project, switch to the root of the source directory, and
type:

    ./gradlew build

On Windows, switch to the root of the source directory, and type:

    gradlew.bat build

The compiled Jar file will be placed in the `build/libs` directory.


## Running

### libfreenect Drivers

To actually connect to the Kinect, you will need to install the 
libfreenect drivers for your platform. Most major Linux distributions
provide compiled binaries in their software repositories. See
[Open Kinect's Getting Started](http://openkinect.org/wiki/Getting_Started)
page for more detailed installation instructions.

### Command Line Help

There are several options available for the program. Running the jar
with the `-h` option will display a helpful description of the options:

    java -jar build/libs/nnclassifier-0.1.jar -h

### Collecting Data 

To take pictures of various objects, you can specify the number of pictures
to take as well as a time delay between successive pictures. For example,
to take 10 pictures:

    java -jar build/libs/nnclassifier-0.1.jar -n 10

To take 10 pictures with a delay of 5 seconds between each:

     java -jar build/libs/nnclassifier-0.1.jar -n 10 -d 5


## Current Status - August 3, 2014

### Operational

- Taking pictures with a delay and saving them as JPG files

### Yet to be Implemented

- The machine learning classifier
- The machine learning predictor


## Third Party Licenses and Attributions

### Apache Commons CLI

This links to the Apache Commons CLI, which is licensed under the 
Apache License, Version 2.0. The license can be downloaded from
http://www.apache.org/licenses/LICENSE-2.0.html. The source code for this
software is available from http://commons.apache.org/cli

### Apache Commons Math

This links to the Apache Commons Math, which is licensed under the
Apache License, Version 2.0. The license can be downloaded from
http://www.apache.org/licenses/LICENSE-2.0.html. The source code for this
software is available from http://commons.apache.org/proper/commons-math

### BoofCV 

This links to BoofCV, which is licensed under the Apache License,
Version 2.0. The license can be downloaded from 
http://www.apache.org/licenses/LICENSE-2.0.txt. The source code for this
software is available from https://github.com/lessthanoptimal/BoofCV
