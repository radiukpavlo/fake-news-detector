#!/bin/bash
set -e

# Запуск FastAPI
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
