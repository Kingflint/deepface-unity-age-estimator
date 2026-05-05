using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using DeepFaceAge.Models;

namespace DeepFaceAge
{
    /// <summary>
    /// Async-friendly client for the DeepFace server. Wraps the bare
    /// <see cref="AgeEstimatorClient"/> with retry, timeout and JSON parsing
    /// helpers. Uses Unity coroutines plus callbacks; consumers can adapt to
    /// async/await with their own helper if running on Unity 2022+.
    /// </summary>
    public class AsyncAgeEstimatorClient
    {
        private readonly AgeEstimatorSettings settings;

        public AsyncAgeEstimatorClient(AgeEstimatorSettings settings)
        {
            this.settings = settings ?? throw new ArgumentNullException(nameof(settings));
        }

        public IEnumerator AnalyzeTexture(Texture2D texture, Action<AnalysisResult> onSuccess, Action<string> onError)
        {
            byte[] png = texture.EncodeToPNG();
            string body = "{\"image\":\"" + Convert.ToBase64String(png) + "\"}";
            yield return PostJson(settings.AnalyzeUri.ToString(), body, onSuccess, onError);
        }

        public IEnumerator SubmitJob(byte[] imageBytes, Action<JobStatus> onSuccess, Action<string> onError)
        {
            string b64 = Convert.ToBase64String(imageBytes);
            string body = "{\"kind\":\"analyze\",\"payload\":{\"image\":\"" + b64 + "\"}}";
            yield return PostJson(settings.JobsUri.ToString(), body, onSuccess, onError);
        }

        public IEnumerator PollJob(string jobId, Action<JobStatus> onSuccess, Action<string> onError)
        {
            string url = $"{settings.JobsUri}/{jobId}";
            using (var req = UnityWebRequest.Get(url))
            {
                ApplyHeaders(req);
                req.timeout = settings.requestTimeoutSeconds;
                yield return req.SendWebRequest();
                HandleResponse(req, onSuccess, onError);
            }
        }

        private IEnumerator PostJson<T>(string url, string body, Action<T> onSuccess, Action<string> onError)
        {
            byte[] payload = Encoding.UTF8.GetBytes(body);
            int attempt = 0;
            int delay = settings.initialBackoffMs;
            while (true)
            {
                attempt++;
                using (var req = new UnityWebRequest(url, "POST"))
                {
                    req.uploadHandler = new UploadHandlerRaw(payload);
                    req.downloadHandler = new DownloadHandlerBuffer();
                    req.SetRequestHeader("Content-Type", "application/json");
                    ApplyHeaders(req);
                    req.timeout = settings.requestTimeoutSeconds;

                    yield return req.SendWebRequest();

                    if (req.result == UnityWebRequest.Result.Success)
                    {
                        var parsed = JsonUtility.FromJson<T>(req.downloadHandler.text);
                        onSuccess?.Invoke(parsed);
                        yield break;
                    }

                    if (attempt > settings.maxRetries)
                    {
                        onError?.Invoke(req.error ?? "request failed");
                        yield break;
                    }
                }

                yield return new WaitForSeconds(delay / 1000.0f);
                delay *= 2;
            }
        }

        private void HandleResponse<T>(UnityWebRequest req, Action<T> onSuccess, Action<string> onError)
        {
            if (req.result == UnityWebRequest.Result.Success)
            {
                var parsed = JsonUtility.FromJson<T>(req.downloadHandler.text);
                onSuccess?.Invoke(parsed);
            }
            else
            {
                onError?.Invoke(req.error ?? "request failed");
            }
        }

        private void ApplyHeaders(UnityWebRequest req)
        {
            if (settings.HasApiKey)
            {
                req.SetRequestHeader("X-API-Key", settings.apiKey);
            }
            if (settings.HasBearer)
            {
                req.SetRequestHeader("Authorization", $"Bearer {settings.bearerToken}");
            }
        }
    }
}
