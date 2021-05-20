FROM python:3.8.5-slim-buster

ADD requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app/
ADD . /app/

EXPOSE 8000

CMD ["python", "manage.py", "makemigrations"]

CMD ["python", "manage.py", "migrate"]

CMD ["gunicorn", "--bind", ":8000", "serving.wsgi"]