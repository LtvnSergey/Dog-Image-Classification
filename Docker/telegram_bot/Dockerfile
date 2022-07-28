#####################################################################################################################
FROM python:3.8

# Mount current directory to /app in the container image
VOLUME ./:app/

# Copy local directory to /app in container
# Dont use COPY * /app/ , * will lead to lose of folder structure in /app
COPY . /app

# Change WORKDIR
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Set exec command
CMD python3 /app/telegram_bot.py

# docker build
# docker build -t "<app name>" .

# docker run

# show all running containers
# docker ps

# Kill and remove running container
# docker rm <containerid> -f

# open bash in a running docker container
# docker exec -ti <containerid> bash

# docker compose
# run and interact between multiple docker containers