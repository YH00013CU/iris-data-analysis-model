FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
EXPOSE 8501

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["bash", "-c", "python api/api.py & streamlit run frontend/iris_app.py --server.enableCORS false --server.port 8501 --server.address 0.0.0.0"]
