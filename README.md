# Face Recognition Attendance System

This is a simple web-based face recognition attendance system implemented using Flask and OpenCV.

## Overview

The system captures real-time video feed from the webcam, detects faces using Haar cascade classifier, and recognizes them based on a pre-trained KNN classifier. Users can add new faces to the system, and the attendance of recognized individuals is logged in a SQL Database.

## Prerequisites

- Python 3.x
- OpenCV
- Flask
- NumPy
- scikit-learn

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/scorpionTaj/face-recognition.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open a web browser and go to `http://localhost:5000/`.

3. Click on the "Prendre la Présence" button to begin capturing attendance.

4. To add a new user, click on the "Ajouter un Nouvel Utilisateur" button and follow the instructions.

## Directory Structure

- `app.py`: Main Flask application file.
- `templates/`: HTML templates for the web interface.
- `static/`: Static files including user images and Trained Model.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar cascade classifier for face detection.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you encounter any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
