# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose ports for Flask (5000) and Streamlit (8501)
EXPOSE 5000
EXPOSE 8501

# Optional: Streamlit needs config to avoid interactive prompts
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run both Flask and Streamlit (supervisord or simple bash)
CMD ["bash", "-c", "python app.py & streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false --server.port 8501 --server.address 0.0.0.0"]
