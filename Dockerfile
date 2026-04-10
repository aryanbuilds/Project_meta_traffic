FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY openenv_requirements.txt /app/openenv_requirements.txt
RUN pip install --no-cache-dir -r /app/openenv_requirements.txt

COPY pyproject.toml /app/pyproject.toml
COPY openenv.yaml /app/openenv.yaml
COPY inference.py /app/inference.py
COPY openenv_selfdriving /app/openenv_selfdriving

RUN chown -R user:user /app
USER user

EXPOSE 8000

CMD ["uvicorn", "openenv_selfdriving.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
