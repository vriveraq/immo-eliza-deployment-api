FROM python:3.11-slim

# Install libgomp (OpenMP runtime) for numpy / sklearn / pycaret
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Avoid .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and artifacts
COPY . .

# Create non-root user for safety
RUN adduser --disabled-password --gecos "" apiuser && \
    chown -R apiuser:apiuser /app

# Switch to non-root user
USER apiuser

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "my_api:app", "--host", "0.0.0.0", "--port", "8000"]