# Базовий образ Node.js
FROM node:22-alpine

# Робоча директорія
WORKDIR /my-app

# Копіюємо package.json та package-lock.json
COPY package*.json ./

# Встановлюємо залежності
RUN npm install

# # 🔹 Додаємо бібліотеки для візуалізації
# RUN npm install react-plotly.js plotly.js

# Копіюємо весь код
COPY . .

# Скрипт для старту
RUN chmod +x start.sh

# Порт для React
EXPOSE 3000

# Команда запуску
CMD ["./start.sh"]
