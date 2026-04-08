FROM python:3.11-slim

WORKDIR /app

COPY openenv_requirements.txt /app/openenv_requirements.txt
RUN pip install --no-cache-dir -r /app/openenv_requirements.txt

COPY openenv_selfdriving /app/openenv_selfdriving
COPY openenv.yaml /app/openenv.yaml

EXPOSE 8000

CMD ["uvicorn", "openenv_selfdriving.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
