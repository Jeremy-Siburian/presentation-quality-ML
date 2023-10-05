# Presentation Quality Assessment Based On Audience Engagement with Open CV and CNN

Project Owners: <br>
[Jeremy Siburian](https://www.linkedin.com/in/jeremy-siburian/), [Addo Bari Alyfathin](https://www.linkedin.com/in/addobari/), Sabir Malik

Last Update: **October 10, 2023**

## Table of Contents

Contents of the documentation are the following: <br>
1. [Project Overview](#1-project-overview)
2. [File Navigation & Program Explanation](#2-file-navigation--program-explanation)
3. [Related Links](#3-related-links)

Note: <br>
This documentation is currently under construction. Therefore, several contents may be missing.

## 1. Project Overview

This codebase was created as part of a team-based project in the Mechanical Engineering Frontiers E course of Waseda University.

Key Contributions: <br>
- Developed a novel presentation scoring algorithm based on the level of audience engagements using machine learning (ML) and neural networks.

- Performed frame extraction, face annotation, and data labeling on a dataset of classroom presentation videos using OpenCV.

- Utilized 3 different ML models (2D CNN, ResNet50, 3D CNN) to extract spatio-temporal features and perform image classification to measure the level of audience engagement.

- Evaluation metrics and cross-comparison between the models showed that a 2D CNN model resulted in a best accuracy of 85%.

For the full details of this project, please refer to the [research paper](https://drive.google.com/file/d/146UNCqA8xWE2SJLHIJXHTwCiZt48eK3D/view?usp=drive_link) here.

## 2. File Navigation & Program Explanation

The main function of each program is described below:

| File Name | Description |
|---|---|
| frame_extraction.py | Source code for extracting frames every 10 seconds from presentation videos. |
| face_detection.py | Source code for detecting and annotating faces in each video frame into Region of Interests (ROIs) using MTCNN face detector. |
| face_labeling.py | Source code for manual labeling of annotated faces based on their attention level. ROIs are then extracted and separated into different directories based on the given label. |
| model_training_2DConv.py | Source code for 2D CNN ML model. |
| model_training_RESNET.py | Source code for pre-trained ResNet50 ML model. |
| model_training_3DConv.py | Source code for 3D CNN ML model. |

## 3. Related Links

**Under construction.**

