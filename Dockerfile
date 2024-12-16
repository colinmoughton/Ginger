# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

COPY requirements.txt /app

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the current directory contents into the container
COPY . /app

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Define the command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
