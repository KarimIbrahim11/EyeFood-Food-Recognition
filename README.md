
# EyeFood (Food-Recognition for Visually Impaired) [THESIS]

### (Demo Video)[https://user-images.githubusercontent.com/47744559/235339233-676e6d52-94e4-428f-b068-b3072bb63795.mp4]

<!--![alt text](https://github.com/KarimIbrahim11/Food-Recognition/blob/main/logo.png?raw=true "Title")-->
<br/><br/>
<div align="left">
  <img src="https://github.com/KarimIbrahim11/Food-Recognition/blob/main/logo.png" width="550" title="EyeFood Logo">
</div>

<br/><br/>

This project is a thesis submission for the degree of Bachelors of Science from Cairo University - Faculty of Engineering. The goal of this project was to develop an food recognition and detection system for visually impaired individuals.

## Description
The system is designed to assist visually impaired individuals in identifying food(including oriental food) in their surroundings through the use of a camera in a smart glasses and machine learning algorithms. The system is able to recognize a wide range of plates (54 dish including both oriental and international dishes) and classify them.

## Features
Object recognition and classification: The system uses deep learning algorithms to recognize and classify objects in real-time.
Audio feedback: The system provides audio feedback to the user, identifying the food plate(s) ahead and their respective locations.
User-friendly interface: The interface used for simulation is kivy running on Raspberry Pi 4B (Raspbian). This is just to simulate the use of smartglasses that embeds a camera.

## Credits
This project was developed by [Mostafa Sherif](https://github.com/Mostafa-Mourad), [Youssef Sayed](https://github.com/youssef998), [Amir Salah](https://github.com/AmirAlahmedy) and [myself](https://github.com/KarimIbrahim11/) under the supervision of [Prof Ibrahim Sobh](https://github.com/IbrahimSobh) and [Prof Ahmed Darwish](https://www.amdarwish.com/). We would like to thank [Valeo Egypt](https://www.valeo.com/en/egypt/) for selecting our project for the 2021 Valeo Mentorship Program.

## Contact
If you have any questions or feedback, please contact [karim-ibrahim](karim.ibrahim.amin@gmail.com) or check the Thesis book and presentation Section at the end of this file.
_____________________________________________________________________________________________________________________________________________________________
## Project Files Structure:

<br/>
`\_Datasets`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`\_Custom Dataset`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`\_food-101`<br/>`
\_master [REPO]`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`\_classification weights`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`\_dataset manipulation`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`\_detection weights`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`\_images`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`\_unsuccessful trials`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`\_visuals`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`classification_inference.py`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`classification_training.py`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`classification_utils.py`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`detection_utils.py`<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;`pipeline.py` 
 
_____________________________________________________________________________________________________________________________________________________________

## Dataset

 The dataset used was based on the [Food101](https://www.kaggle.com/datasets/dansbecker/food-101) Dataset which is a balanced ds that has a total of 101K images (1000 image per 101 Classes). DS processing process involved excluding unpopular dishes from the food-101 dataset and collecting/adding oriental dishes to the unexcluded classes. A total of 54 Class was included in the final Dataset:  

![ds_1](https://user-images.githubusercontent.com/47744559/235338365-c049c882-f405-44ea-91ab-441f28271a17.jpg)
![ds2](https://user-images.githubusercontent.com/47744559/235338369-1f36f95e-7f2c-4b42-96f4-8396305f720c.jpg)

## Pipeline and Training
  
### Pipeline:
  A ```FasterRCNN``` object detector is used to identify the bounding blocks of the plates of food (if any) and output them to a ```MobileNetV2``` classifier that is trained on the aforementioned Custom Dataset. The output is the location of each bounding box and the predicted label for that box. 
### Methodology: 
  The Classifier was Fine-tuned on the 54k image dataset for 70 epochs showing the following:

### - Training vs Val Accuracy

![model_accuracy](https://user-images.githubusercontent.com/47744559/235338559-8c015e09-c727-46ef-a3b9-617a63b84c5d.png)

### - Confusion Matrix and Classification Report

![54_confusion_matrix](https://user-images.githubusercontent.com/47744559/235338568-aa7c3f50-6bcd-4e14-9756-d7591b22f82d.png)

![54 Classification Report](https://user-images.githubusercontent.com/47744559/235338685-eeac9fbd-404a-4409-8e55-8a0ebff583d0.png)

## Sample Results

### - Full Image (Object Detector output)
![Full Image](https://user-images.githubusercontent.com/47744559/235338752-75859e14-df1c-4e57-bb55-243308b18c68.jpg)
### - Predicted Plates (Classifier output)
![Screenshot 2022-07-30 140009](https://user-images.githubusercontent.com/47744559/235338754-dde16d48-7b93-44a2-a03b-381f0d34f229.jpg)
![Screenshot 2022-07-30 140021](https://user-images.githubusercontent.com/47744559/235338755-62533c8d-2575-48e2-a4b8-5d6dbeb43a38.jpg)
![Screenshot 2022-07-30 140032](https://user-images.githubusercontent.com/47744559/235338756-fb49270c-9264-4fdd-9156-c57fba4b6511.jpg)
![Screenshot 2022-07-30 140044](https://user-images.githubusercontent.com/47744559/235338757-50617416-a5ef-4cba-a6e9-2bd8db9871c2.jpg)

### Thesis book and presentation: 
- [Thesis Book](https://github.com/KarimIbrahim11/EyeFood-Food-Recognition/files/11360711/Group8_Graduation_Project_Eye-Book-Final.pdf) 
- [Thesis Presentation](https://github.com/KarimIbrahim11/EyeFood-Food-Recognition/files/11360708/EyeFood_Final.pptx)



 
