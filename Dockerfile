FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y g++ unixodbc-dev

WORKDIR /airbnb_prediction
COPY . .

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
