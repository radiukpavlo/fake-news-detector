# Базовий образ .NET SDK для збірки
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Копіюємо solution файл і csproj у контейнер
# COPY RealFakeNews/*.csproj ./RealFakeNews/
COPY RealFakeNews/RealFakeNews.sln ./RealFakeNews.sln
COPY RealFakeNews/RealFakeNews/RealFakeNews.csproj ./RealFakeNews/RealFakeNews.csproj

# Робимо restore
RUN dotnet restore ./RealFakeNews/RealFakeNews.csproj

# Копіюємо решту коду
COPY RealFakeNews/ ./RealFakeNews/

WORKDIR /src/RealFakeNews
RUN dotnet publish -c Release -o /app/publish

# Створюємо runtime образ
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .

# Скрипт запуску
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 5000
CMD ["./start.sh"]

ENTRYPOINT ["dotnet", "RealFakeNews.dll"]
