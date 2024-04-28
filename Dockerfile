FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt  # Install OpenCV, Numpy, and other dependencies

COPY . .

# Replace with your model file path if needed
COPY 1.tflite . 

CMD ["flask", "run"]  # Run Flask app using the command