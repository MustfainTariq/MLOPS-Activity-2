# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir Flask scikit-learn numpy

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run your app
CMD ["python", "app.py"]
