using Microsoft.AspNetCore.Mvc;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;

namespace RealFakeNews.Controllers;

[ApiController]
[Route("api/ml")]
public class MLController : ControllerBase
{
    private readonly HttpClient _http;

    public MLController(IHttpClientFactory httpClientFactory)
    {
        _http = httpClientFactory.CreateClient("MlService");
    }

    private static async Task<IActionResult> ForwardResponse(HttpResponseMessage response)
    {
        using (response)
        {
            var payload = await response.Content.ReadAsStringAsync();
            var contentType = response.Content.Headers.ContentType?.ToString() ?? "application/json";

            return new ContentResult
            {
                StatusCode = (int)response.StatusCode,
                Content = payload,
                ContentType = contentType
            };
        }
    }

    [HttpGet("health")]
    public async Task<IActionResult> Health(CancellationToken cancellationToken)
    {
        var response = await _http.GetAsync("health", cancellationToken);
        return await ForwardResponse(response);
    }

    public record TrainRequest(string ModelName, bool DownloadDataset = false, bool ForceRebuild = false);

    [HttpPost("train")]
    public async Task<IActionResult> Train([FromBody] TrainRequest? request, CancellationToken cancellationToken)
    {
        request ??= new TrainRequest("roberta-base");

        var response = await _http.PostAsJsonAsync(
            "train",
            new
            {
                model_name = request.ModelName,
                download_dataset = request.DownloadDataset,
                force_rebuild = request.ForceRebuild
            },
            cancellationToken);

        return await ForwardResponse(response);
    }

    [HttpGet("train/status/{modelName}")]
    public async Task<IActionResult> TrainingStatus(string modelName, CancellationToken cancellationToken)
    {
        var response = await _http.GetAsync($"train/status/{modelName}", cancellationToken);
        return await ForwardResponse(response);
    }

    public record PredictRequest(string ModelName, string Text);

    [HttpPost("predict")]
    public async Task<IActionResult> Predict([FromBody] PredictRequest? request, CancellationToken cancellationToken)
    {
        request ??= new PredictRequest("roberta-base", string.Empty);

        var response = await _http.PostAsJsonAsync(
            "predict",
            new
            {
                model_name = request.ModelName,
                text = request.Text
            },
            cancellationToken);

        return await ForwardResponse(response);
    }

    [HttpGet("metrics/{modelName}")]
    public async Task<IActionResult> Metrics(string modelName, CancellationToken cancellationToken)
    {
        var response = await _http.GetAsync($"metrics/{modelName}", cancellationToken);
        return await ForwardResponse(response);
    }

    [HttpGet("metrics/{modelName}/plots/{plotName}")]
    public async Task<IActionResult> MetricPlot(string modelName, string plotName, CancellationToken cancellationToken)
    {
        var response = await _http.GetAsync($"metrics/plots/{modelName}/{plotName}", cancellationToken);
        if (response.IsSuccessStatusCode)
        {
            var contentType = response.Content.Headers.ContentType?.ToString() ?? "application/octet-stream";
            var bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
            response.Dispose();
            return File(bytes, contentType, plotName);
        }

        return await ForwardResponse(response);
    }

    [HttpGet("models")]
    public async Task<IActionResult> Models(CancellationToken cancellationToken)
    {
        var response = await _http.GetAsync("models", cancellationToken);
        return await ForwardResponse(response);
    }
}
