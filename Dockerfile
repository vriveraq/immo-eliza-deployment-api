FROM python:3.11-slim

# Avoid .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code (API, models, preprocessing, etc.)
COPY . .


# Create non-root user for safety purposes
RUN adduser --disabled-password --gecos "" apiuser

# Give directiry ownership to the new user
RUN chown -R apiuser:apiuser /app

# change user
USER apiuser


# Expose FastAPI port
EXPOSE 8000

# Start FastAPI 
CMD ["uvicorn", "my_api:app", "--host", "0.0.0.0", "--port", "8000"]