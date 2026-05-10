FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY sim/ ./sim/
COPY policies/ ./policies/

# Expose port
EXPOSE 8000

# Command to run
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
