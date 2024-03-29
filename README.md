# Facial Recognition 3

This project demonstrates a simple web application that allows users to upload an image and detect facial features such as eyes, nose, and mouth using OpenCV.

## Requirements

    Python 3.x
    Flask
    OpenCV
    Numpy
    Werkzeug

## Usage
1. Clone the repo
   ```
   git clone https://github.com/0vr/FR3.git
   ```
2. Install the required packages:
   ```
   pip instal -r requirements.txt
   ```
3. Run the app
   ```
   python app.py
   ```
4. Open your web browser and navigate to http://localhost:5000 to use the application.

## Todo
- [ ] Fix some scanning issues
- [ ] Better UI/UX
- [ ] Make the rectangles editable / edit the facial feature boxes
- [ ] Implement a database to store the original face and facial features
- [ ] Add error handling
- [ ] Add tests for functions
- [ ] API to retrieve the images (store by UUID?)
