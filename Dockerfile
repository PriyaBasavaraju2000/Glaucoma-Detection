FROM python:3.8-slim

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "120"]
