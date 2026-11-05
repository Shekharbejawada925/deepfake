FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home appuser && mkdir -p /app/data/runtime && chown -R appuser:appuser /app
USER appuser
ENV HOST=0.0.0.0
EXPOSE 5000
CMD ["python", "app.py"]
