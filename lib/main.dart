import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
//import 'package:tflite/tflite.dart';
import 'package:tensorflowexjobb/tensorflowexjobb.dart';
import "dart:typed_data";
import 'dart:developer';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: TfliteHome(),
    );
  }
}

class TfliteHome extends StatefulWidget {
  @override
  _TfliteHomeState createState() => _TfliteHomeState();
}

class _TfliteHomeState extends State<TfliteHome> {
  File _file;
  int _numOfPredictions = 100;
  
  List _recognitions;

  @override
  void initState() {
    super.initState();
    runExperiment();  
  }

void runExperiment() async {
  await loadModel();
  for (int i = 1; i <= _numOfPredictions; i++) {
      
      await prepareImage(i);
      log("time : hello");
      await getImageProcessing();
      log("time : hello");
      await getImage();
      log("time : hello");
  } 
  exit(0);
}

// Loads float point mobilnet model, expects 224x224 images
  loadModel() async {
    Tensorflowexjobb.close();
    try { 
      String res;
      res = await Tensorflowexjobb.loadModel(
        model: "assets/tflite/mobilenet_v1_1.0_224.tflite",
        labels: "assets/tflite/mobilenet_v1_1.0_224.txt",
        numThreads: 1,
      );
    } on PlatformException {
    }
  }

// Prepares the image by getting path
  prepareImage(int iteration) async {
    ByteData data = await rootBundle.load('assets/images/image_00$iteration.jpg');
    String directory = (await getTemporaryDirectory()).path;
    File file = await writeToFile(data, '$directory/image_$iteration.jpg');
    
    setState(() {
      _file = file;
    });
  }

// Function called when pressing buttons, starts timer and does set amount of inferences.
  Future getImage() async {
    await predictImageClassification();
    setState(() {
    });

  }

getImageProcessing() async {
  await Tensorflowexjobb.loadImageBitmap(
    path: _file.path,
    imageMean: 127.5,
    imageStd: 127.5, 
    asynch: false,
  );
}
// Image inference, does ImageProcess (bitmap to ByteBuffer) and tflite inference.
  predictImageClassification() async {
    var recognitions = await Tensorflowexjobb.runModelOnImage(
      path: _file.path,
      numResults: 6,
      threshold: 0.05,
      imageMean: 127.5,
      imageStd: 127.5,
      asynch: false,
    );

    setState(() {
      _recognitions = recognitions;
    });
  }

Future<File> writeToFile(ByteData data, String path) {
  ByteBuffer buffer = data.buffer;
  return File(path).writeAsBytes(buffer.asUint8List(
    data.offsetInBytes,
    data.lengthInBytes,
  ));

}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
    );
  }
}
