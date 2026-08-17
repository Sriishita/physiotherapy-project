# 🏥 PhysioVision

### AI-Powered Physiotherapy & Rehabilitation Assistant

**PhysioVision** is an AI-based physiotherapy assistance system developed during our **Infosys Internship**. The project uses **Computer Vision, MediaPipe, and Machine Learning** to analyze physiotherapy exercises and provide feedback on body posture and movement.

The project was developed with the goal of exploring how AI can assist users in performing rehabilitation exercises more effectively.

---

## 📌 About the Project

PhysioVision analyzes a user's movements while performing physiotherapy exercises and evaluates their posture and form.

We created our **own dataset** consisting of physiotherapy exercise samples and used **MediaPipe Pose** to detect and track body landmarks. These landmarks were then used to analyze body movements and identify exercise patterns.

The complete system was integrated into a simple interactive web interface using **Streamlit**, allowing users to interact with the AI-based exercise analysis system.

---

## 🔄 How It Works

```text
User performs physiotherapy exercise
                ↓
        Image / Video Input
                ↓
        MediaPipe Pose
                ↓
      Body Landmark Detection
                ↓
      Feature / Angle Analysis
                ↓
       Exercise Evaluation
                ↓
       Feedback to the User
```

### 1. Dataset Creation

Instead of relying entirely on an existing dataset, we **created and collected our own dataset** containing samples of different physiotherapy exercises.

The dataset was prepared and organized according to the exercises that the system was designed to analyze.

### 2. Pose Detection

We used **MediaPipe Pose** to detect important body landmarks from the input.

The detected landmarks include points around the:

* Shoulders
* Elbows
* Wrists
* Hips
* Knees
* Ankles

These landmarks allow the system to understand the user's body position and movement.

### 3. Movement Analysis

The detected landmarks were processed to derive useful movement features such as **joint positions, angles, and body alignment**.

These features were then used for exercise recognition and movement evaluation.

### 4. AI/ML Processing

Machine learning techniques were applied to the extracted features to analyze the exercises and determine whether the user's movement followed the expected pattern.

### 5. Web Application

We developed an interactive interface using **Streamlit**, allowing users to provide their exercise input and view the analysis and feedback through a simple web interface.

---

## 🛠️ Tech Stack

**Programming:**
Python

**AI / ML:**
Machine Learning, Deep Learning, Computer Vision

**Pose Estimation:**
MediaPipe Pose

**Image & Video Processing:**
OpenCV

**Data Processing:**
NumPy, Pandas

**Web Application:**
Streamlit

**Development & Collaboration:**
Git, GitHub, Jupyter Notebook, VS Code

---

## ✨ Key Features

* 🧍 Human pose and landmark detection
* 🦴 Body movement and joint-angle analysis
* 🏃 Physiotherapy exercise recognition
* 📹 Exercise video/image analysis
* 🤖 AI-based movement evaluation
* 💡 Exercise feedback
* 🌐 Interactive Streamlit web interface
* 📊 Custom-built physiotherapy dataset

---

## 🖥️ Web Application

The project includes a **Streamlit-based web application** that provides an easy-to-use interface for interacting with the physiotherapy analysis system.

Users can provide their exercise input and receive the corresponding movement analysis and feedback through the application.

> 📸 *Add screenshots of your Streamlit interface here.*

---

## 📂 Project Workflow

```text
Custom Dataset
      ↓
Data Preprocessing
      ↓
MediaPipe Pose Detection
      ↓
Landmark Extraction
      ↓
Feature & Angle Calculation
      ↓
Machine Learning Model
      ↓
Exercise Evaluation
      ↓
Streamlit Web Application
```

---

## 👩‍💻 My Contribution

As part of the project team, I contributed to the development of the **AI/ML and computer vision components** of the project.

My work included:

* Working with the custom physiotherapy dataset
* Exploring computer vision and pose estimation
* Working with **MediaPipe Pose**
* Processing body landmark data
* Contributing to movement and exercise analysis
* Working with the machine learning pipeline
* Collaborating with the team using **Git and GitHub**

---

## 🚀 Future Scope

Some potential improvements for the project include:

* Real-time exercise monitoring
* Support for more physiotherapy exercises
* Improved movement accuracy
* Automatic repetition counting
* Range-of-motion analysis
* Personalized exercise recommendations
* Mobile application integration
* Progress tracking for users

---


## ⚠️ Disclaimer

PhysioVision is an educational and research-oriented project and is **not a substitute for professional medical or physiotherapy advice**.

---

### ⭐ Built with Python, Computer Vision & AI

Developed as part of the **Infosys Internship**.
