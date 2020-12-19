# csharp-sample-app

**Build Status**

[![Build status](https://ci.appveyor.com/api/projects/status/97yet8bxbbei2hpe?svg=true)](https://ci.appveyor.com/project/umangmehta12/csharp-sample-apps)

**csharp-sample-app** is a windows application that demonstrates the use of the Affdex SDK for Windows. This app uses the [PhotoDetector](http://developer.affectiva.com/pages/platforms/v3_1/windows/classdocs/Affdex/html/1bdd6e83-b415-70d3-5b67-2697a768b717.htm), [VideoDetector](http://developer.affectiva.com/pages/platforms/v3_1/windows/classdocs/Affdex/html/6e4b1996-68bf-4750-439a-731c2be17537.htm) and [CameraDetector](http://developer.affectiva.com/pages/platforms/v3_1/windows/classdocs/Affdex/html/1d7a795f-92f8-e0e5-f48a-79d1d1941091.htm). It can analyze live camera feed, photos and videos.

It runs on Windows 7.0, 8.0, 8.1 and 10

#### To build this project from source, you will need:

*   Visual Studio 2015

*   To [download and install the Windows SDK (64-bit)](http://developer.affectiva.com/downloads) from Affectiva.

    By default, the Windows SDK is installed to the following location: ```C:\Program Files\Affectiva\Affdex SDK```

    The sample app requires you to point to the data directory which is located at : ```C:\Program Files\Affectiva\Affdex SDK\data```.
    
    Here is the help text to get started:

```

-i, --input       Required. Input stream to be processed. Set the input to
                    "camera" if you want to use the live camera stream.

 -d, --data        Required. Data Directory.

 -f, --numFaces    (Default: 1) Set Max Number of faces.

 -m, --faceMode    (Default: 0) Set face detector mode.

 --help            Display this help screen.

```
*   The csharp-sample app using CommanLine Parser Utility which can be managed by NuGet.

*   Build the project

*   Run the app through Visual Studio

**Note** It is important not to mix Release and Debug versions of the DLLs. If you run into issues when switching between the two different build types, check to make sure that your system path points to the matching build type.

Copyright (c) 2016 Affectiva. All rights reserved.
