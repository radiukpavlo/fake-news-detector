var builder = WebApplication.CreateBuilder(args);

// ✅ CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowFrontend",
        policy =>
        {
            policy.WithOrigins("http://localhost:3000")
                  .AllowAnyMethod()
                  .AllowAnyHeader();
        });
});

builder.Services.AddControllers();
builder.Services.AddHttpClient();

var app = builder.Build();

// ✅ CORS перед MapControllers
app.UseRouting();
app.UseCors("AllowFrontend");

app.UseAuthorization();
app.MapControllers();

app.Run();
