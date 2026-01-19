FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001 7860

CMD ["sh", "-c", "python -m app.api.flask_app & sleep 5 && python -m app.ui.gradio_app"]
