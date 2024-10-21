# Use the official Python 3.10.9 image
FROM python:3.10.9

# Update package list and install dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install requirements.txt 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# create a new user named 'user'
RUN useradd user

# switch the user to 'user'
USER user

# set the working directory to user's home director
WORKDIR $HOME/app

# copy the current directory contents into the container at $HOME/app setting to 
COPY --chown=user . $HOME/app

#RUN mkdir -p /app/cache && chmod -R 777 /app/cache 
#RUN mkdir -p ./85bbcb92-49d4-4d55-a62e-ddbfab54d4a2 && chmod -R 777 ./85bbcb92-49d4-4d55-a62e-ddbfab54d4a2

ENV TRANSFORMERS_CACHE=/app/cache

# Start the FastAPI app on port 7860, the default port expected by Spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]