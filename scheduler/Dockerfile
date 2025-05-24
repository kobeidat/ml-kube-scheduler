FROM python:3.10-slim
WORKDIR /app
COPY scheduler.py /app/
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "scheduler.py"]
