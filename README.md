# Dog-Image-Classification

<img src="https://github.com/LtvnSergey/Dog-Image-Classification/blob/main/assets/images/meme.png" width="500">

## PROJECT OVERVIEW
Image classification of dog breeds.

Project goals:
  1. Train neural model using Pytorch library and dataset with dog images 'Imagewoof' from [imagenette](https://github.com/fastai/imagenette)
  2. Build Flask app 
  3. Create Docker images
  4. Create Telegram bot for image classification

### 1. Training

- For training was used convolutional pre-trained Resnet50 model with addtional two linear layers at the end 
- Pre-trained weights were frozen during the training and only final additional layers were updated
- As loss function cross entropy loss was used
- 10 epochs of training were performed
- Quality of the model prediciton was measured by accuracy and confusion matrix
- Overlall accuracy on validation set exceeds 0.9 


- More information about data analysis and training process can be found in notebook:
[Imagewoof.ipynb](https://github.com/LtvnSergey/Dog-Image-Classification/blob/main/notebooks/Imagewoof.ipynb)

- Trained model:
[model_v1.pt](https://github.com/LtvnSergey/Dog-Image-Classification/blob/main/models/model_v1.pt)


### 2. Flask app

- To run Flask web app type the command:
``` pytho3 app.py ```
- Go to ``` http://127.0.0.1:5000 ``` in browser
- Upload image with a link or with Choose file option:

<img src="https://github.com/LtvnSergey/Dog-Image-Classification/blob/main/assets/images/home_page.png" width="500">

- Push Upload button and get 3 top predictions ranked by certainty:

<img src="https://github.com/LtvnSergey/Dog-Image-Classification/blob/main/assets/images/prediction_page.png" width="500">


### 3. Docker

- Install Docker: [docker](https://docs.docker.com/get-docker/)


- Download Docker images from docker repository

  a. Flask App docker image: 
``` docker pull plasticglass/telegram_bot:latest ```

  b. Telegram bot docker image: 
``` docker pull plasticglass/image_app:latest ```


- (Optionaly) Build Docker images:
  
  a. Flask App docker image: 
``` docker build -t image_app -f Docker/flask_app/Dockerfile ```

  b. Telegram bot docker image: 
``` docker build -t bot_app -f Docker/telegram_app/Dockerfile ```
 

- Run following commands to execute docker container:

  a. Flask App docker container: 
```  docker run -p 5000:3000 image_app ```

  b. Telegram bot docker container: 
``` docker run bot_app ```


### 4. Telegram bot

Telegram bot allows to classify images using telegram, 
to do so follow the steps:

- Enable telegram bot:
   
  a. Run in terminal: 
``` python3 telegram_bot.py ```

  b. Or run a docker container: 
``` docker run bot_app ```

- Go to http://t.me/DogClassBot in telegram app
  
<img src="https://github.com/LtvnSergey/Dog-Image-Classification/blob/main/assets/images/telegram_start.png" width="500">

- Upload image to bot to get prediction 

<img src="https://github.com/LtvnSergey/Dog-Image-Classification/blob/main/assets/images/telegram_prediction.png" width="500">


## 5. Moduls and tools

#### Web Development:
Flask | HTML | CSS | Docker | Telegram

#### Python - CNN:
Pytorch | Numpy | Pandas | Torchvision | Pillow

 
