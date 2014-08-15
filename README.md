# ML Deer Detector

[![Build Status](https://travis-ci.org/craigthomas/DeerDetector.svg?branch=master)](https://travis-ci.org/craigthomas/MLDeerDetector) [![Coverage Status](https://coveralls.io/repos/craigthomas/DeerDetector/badge.png)](https://coveralls.io/r/craigthomas/DeerDetector)


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

    java -jar build/libs/visualclassifier-0.1.jar -h

### Collecting Data 

The `collect` sub-command allows to you capture images for training.
To take pictures of various objects, you can specify the number of pictures
to take as well as a time delay between successive pictures. For example,
to take 10 pictures:

    java -jar build/libs/visualclassifier-0.1.jar collect -n 10

To take 10 pictures with a delay of 5 seconds between each:

    java -jar build/libs/visualclassifier-0.1.jar collect -n 10 -d 5

To take a picture using the infrared camera:

    java -jar build/libs/visualclassifier-0.1.jar collect -i

Saving files to a different path:

    java -jar build/libs/visualclassifier-0.1.jar collect -p /path/to/save/to


## Current Status - August 7, 2014

### Operational

- Taking pictures with a delay and saving them as JPG files
- Taking pictures using the IR camera and saving them as JPG files

### Yet to be Implemented

- The machine learning classifier
- The machine learning predictor


## Third Party Licenses and Attributions

### args4j

The project makes use of the args4j command-line argument parser,
which is available at https://github.com/kohsuke/args4j and is
licensed under the following terms:

    Copyright (c) 2013 Kohsuke Kawaguchi and other contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:
 
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

### JBlas

This project makes use of the jblas linear algebra library, which is
available at https://github.com/mikiobraun/jblas and is licensed under 
the following terms:

    Copyright (c) 2009, Mikio L. Braun and contributors
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
    
        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials provided
          with the distribution.
    
        * Neither the name of the Technische Universität Berlin nor the
          names of its contributors may be used to endorse or promote
          products derived from this software without specific prior
          written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### BoofCV 

This project makes use of BoofCV, which is licensed under the Apache License,
Version 2.0. The license can be downloaded from 
http://www.apache.org/licenses/LICENSE-2.0.txt. The source code for this
software is available from https://github.com/lessthanoptimal/BoofCV
