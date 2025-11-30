FROM python:3.7
ENV PYTHONUNBUFFERED=1

RUN pip install kubernetes numpy
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY custom-scheduler.py /custom-scheduler.py
COPY prometheus.py /prometheus.py
COPY config.py /config.py
COPY model.py /model.py
CMD ["python", "custom-scheduler.py"]
