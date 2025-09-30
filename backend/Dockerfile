FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

COPY RealFakeNews/RealFakeNews.sln ./
COPY RealFakeNews/RealFakeNews/RealFakeNews.csproj ./RealFakeNews/
RUN dotnet restore ./RealFakeNews/RealFakeNews.csproj

COPY RealFakeNews/ ./RealFakeNews/
WORKDIR /src/RealFakeNews
RUN dotnet publish -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .

EXPOSE 5000
ENTRYPOINT ["dotnet", "RealFakeNews.dll"]
