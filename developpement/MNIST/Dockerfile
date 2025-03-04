# Use an official Python runtime as the parent image
FROM nvidia/cuda:12.2.2-base-ubuntu20.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Set Python3 as the default Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Run mnist_api.py when the container launches
CMD ["python", "mnist_api.py", "--model_path", "weights/mnist.pth"]
