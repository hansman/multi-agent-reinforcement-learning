FROM python:3.5

# Install app
COPY ./gym-http-api /app/
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "gym_http_server.py", "-l", "0.0.0.0"]
