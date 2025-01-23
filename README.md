# Road Monitor: Intelligent System for Truck and Plate Detection

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)

![Road Monitor Banner](https://ibb.co/98yDH4L) 

## 📚 Overview

**Road Monitor** is our graduation project developed under the supervision of **Dr. Bader Mohammed Albashar** at **Imam Mohammad Ibn Saud Islamic University**. This intelligent system is designed to detect large trucks and recognize their license plates in real-time, addressing the pressing issue of traffic congestion in Riyadh caused by unauthorized truck movements during restricted hours.

## 👥 Team

- **Mohammed Muflih Almuflih** (Main Developer & Team Leader)
- **Abdulrahman Abdulaziz Alsanad** 
- **Muhannad Abdulaziz Alosaili** 

## 🚀 Features

- **Real-Time Truck Detection:** Utilizes YOLOv8 for accurate and speedy detection of large trucks.
- **Bilingual License Plate Recognition:** Employs Azure OCR to recognize both Arabic and English characters on license plates.
- **Robust Performance:** Handles challenging environmental conditions such as low visibility, rain, and glare.
- **Data Processing:** Includes modules for adding missing data and visualizing detection results.
- **Modular Design:** Structured into distinct modules for easy maintenance and scalability.

## 🛠️ Technologies & Libraries

- **Programming Language:** Python 3.9
- **Object Detection:** [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics)
- **OCR:** [Azure Cognitive Services OCR](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/)
- **Image Processing:** [OpenCV](https://opencv.org/)
- **Data Handling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Tracking:** [FilterPy](https://github.com/rlabbe/filterpy) (SORT Algorithm)
- **Additional Libraries:** Requests, Python-Bidi, SciPy and more (check requirement.txt)

## 📦 Installation
- Create a Virtual Environment
- Install All Dependencies (you can find them inside requirements.txt)
- Configure Azure OCR (Setup your API key and endpoint)

## 🏃‍♂️ Usage
run **main.py** then **add_missing_data.py** and finally **visualize.py**

## 📫 Contact
For any inquiries or contributions, please contact me @ mmuflihofficial@gmail.com

### Clone the Repository

```bash
git clone https://github.com/yourusername/road-monitor-system.git
cd road-monitor-system
