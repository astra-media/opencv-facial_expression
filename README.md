# Machine Learning (OpenCV) - Facial Expressions

## About this Project

This machine learning project, developed as part of my capstone project in 2018, utilizes Python, OpenCV (Open Source Computer Vision Library), and face datasets to train a computer to recognize human facial expressions. The goal was to classify emotions into four categories: "angry," "happy," "sad," and "neutral," and detect these expressions in real time.

In addition to using OpenCV, I integrated TouchDesigner by Derivative, a visual programming language, to create real-time interactive animations based on facial expressions. The trained model sends data through a socket (networking interface), which TouchDesigner then uses to generate interactive visuals corresponding to each emotion.

## Install dependencies

```
$ pip install -r 'recogniser_trainer/requirements.txt'
```

## Operation

Once all required modules are installed, the following commands can be used:

```
$ 'recogniser_trainer\python emotionrecogniser.py --update' to train the model.
$ 'recogniser_trainer\python emotionrecogniser.py' to detect facial expressions in real time.
$ 'interactive_animation_2024.toe' for real-time interactive animation.
```

The code was later modified to be compatible with the latest versions of Python and OpenCV. This project was inspired by Paul van Gent's "Making an Emotion-Aware Music Player."

## Technical Sheet

The following technologies were utilized in this project:

- Python 3.12.4
- Modules:
  - opencv-python 4.10.0.84
  - opencv-contrib-python 4.10.0.84
  - numpy 2.0.1
- TouchDesigner by Derivative

## Reference:

van Gent, P. (2016). Emotion Recognition With Python, OpenCV, and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from: Paul van Gent's blog
