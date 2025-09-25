#!/bin/sh

# Очікуємо готовність ml-service
echo "Waiting for ml-service..."
while ! nc -z ml-service 8000; do
  sleep 1
done
echo "ml-service is up!"

dotnet RealFakeNews.dll
