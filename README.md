# ✋ Sign Language Detection System

## 👩‍💻 Team Members

| Name | Roll No |
|------|----------|
| Kodali Naga Sreeja | 2410030076 |
| Bijju Adithi Yadav | 2410030228 |
| Likitha Thumma | 2410030229 |
| H. Esha Manogna | 2410030232 |
| Malreddy Manogna | 2410030430 |

---

## 📘 Overview

The **Sign Language Detection System** is a machine learning-based application that detects and recognizes hand gestures from sign language.  
It uses **MediaPipe** for detecting hand landmarks and a **Random Forest Classifier** model trained on extracted features to predict the corresponding alphabet or word.  

The system includes a simple **Flask-based web interface** that allows users to upload sign videos and get instant predictions.

---

## 🎯 Purpose

- To assist communication for people with hearing or speech impairments.  
- To demonstrate practical applications of **computer vision** and **machine learning**.  
- To provide a basic foundation for real-time sign language recognition systems.  

---

## ⚙️ How It Works

1. The user uploads a **sign language video** through the web interface.  
2. The system extracts **hand landmarks** using MediaPipe.  
3. The trained **machine learning model** analyzes the landmarks.  
4. The predicted letter or word is displayed on the webpage.  

---

## 🧠 Technologies Used

- **Python**  
- **Flask** – web framework  
- **MediaPipe** – hand tracking and landmark detection  
- **OpenCV** – video frame processing  
- **Scikit-learn** – model training (Random Forest Classifier)  
- **Pandas, NumPy** – data handling  
- **HTML, CSS** – front-end design  

---

## ✅ Key Features

- Upload and predict sign gesture from video.  
- Uses **hand landmarks** for recognition (no special sensors).  
- Option to **train model** from collected gesture data.  
- Simple and lightweight Flask web interface.  
- Works completely **offline** after model is trained.  

