import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:path_provider/path_provider.dart';

class StaticImage extends StatefulWidget {
  @override
  _StaticImageState createState() => _StaticImageState();
}

class _StaticImageState extends State<StaticImage> {
  File _image;
  bool _busy;
  Image image;
  String label;

  final picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _busy = true;
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;

    List<Widget> stackChildren = [];

    stackChildren.add(
      Positioned(
          // using ternary operator
          child: _image == null
              ? Container(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      Text("Please Select an Image"),
                    ],
                  ),
                )
              : // if not null then
              Container(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      Text(label),
                      image //Image.file(_image)
                    ],
                  ),
                )),
    );

    return Scaffold(
      appBar: AppBar(
        title: Text("Food Detector"),
      ),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: <Widget>[
          FloatingActionButton(
            heroTag: "Fltbtn2",
            child: Icon(Icons.camera_alt),
            onPressed: getImageFromCamera,
          ),
          SizedBox(
            width: 10,
          ),
          FloatingActionButton(
            heroTag: "Fltbtn1",
            child: Icon(Icons.photo),
            onPressed: getImageFromGallery,
          ),
        ],
      ),
      body: Container(
        alignment: Alignment.center,
        child: Stack(
          children: stackChildren,
        ),
      ),
    );
  }

  // gets image from camera and runs detectObject
  var url = 'http://192.168.1.7:5000/test';

  Future getImageFromCamera() async {
    final pickedFile = await picker.getImage(source: ImageSource.camera);

    // setState(() {
    if (pickedFile != null) {
      _image = File(pickedFile.path);
      print("YAAY AN IMAGE IS SELECTED");
      var request = new http.MultipartRequest("POST", Uri.parse('$url'));
      request.files.add(await http.MultipartFile.fromPath(
        'image',
        _image.path,
        contentType: new MediaType('application', 'jpeg'),
      ));
      http.Response response1 =
          await http.Response.fromStream(await request.send());
      print("Result: ${response1.statusCode}");
      print(response1.body);
      setState(() {
        _image = File(pickedFile.path);
        image = Image.memory(response1.bodyBytes);
        label = response1.headers['lp'];
      });
    } else {
      print("No image Selected");
    }
    // detectObject(_image);
  }

  Future getImageFromGallery() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      _image = File(pickedFile.path);
      print("YAAY AN IMAGE IS SELECTED");
      var request = new http.MultipartRequest("POST", Uri.parse('$url'));
      request.files.add(await http.MultipartFile.fromPath(
        'image',
        _image.path,
        contentType: new MediaType('application', 'jpeg'),
      ));
      http.Response response1 =
          await http.Response.fromStream(await request.send());
      print("Result: ${response1.statusCode}");
      print(response1.body);
      setState(() {
        _image = File(pickedFile.path);
        image = Image.memory(response1.bodyBytes);
        label = response1.headers['lp'];
      });
    } else {
      print("No image Selected");
    }
    // detectObject(_image);
  }
}
