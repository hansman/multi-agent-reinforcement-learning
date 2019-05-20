FROM python:3.5

# Install app
COPY ./gym-http-api /app/
WORKDIR /app

RUN pip install box2d-py && \
  pip install scipy && \
  pip install --requirement requirements.txt

EXPOSE 5000

CMD ["python3", "gym_http_server.py", "--listen", "0.0.0.0"]
