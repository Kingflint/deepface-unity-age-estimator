using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace DeepFaceAge
{
    /// <summary>
    /// Lightweight client for the DeepFace Age Estimator HTTP service.
    /// Drop this script on any GameObject and call <see cref="Analyze"/>.
    /// </summary>
    [Serializable]
    public class AnalyzeResponse
    {
        public string error;
        public string code;
        public string raw_json;
    }

    public class AgeEstimatorClient : MonoBehaviour
    {
        [Tooltip("Base URL of the DeepFace API, including scheme.")]
        public string baseUrl = "http://localhost:5000";

        [Tooltip("Optional API key, sent as X-API-Key.")]
        public string apiKey = "";

        [Tooltip("Request timeout in seconds.")]
        public int timeoutSeconds = 30;

        public IEnumerator Analyze(Texture2D texture, Action<AnalyzeResponse> onComplete)
        {
            if (texture == null)
            {
                onComplete?.Invoke(new AnalyzeResponse { error = "texture is null", code = "client_error" });
                yield break;
            }

            byte[] png = texture.EncodeToPNG();
            string b64 = Convert.ToBase64String(png);
            string body = "{\"image\":\"" + b64 + "\"}";

            using (var req = new UnityWebRequest($"{baseUrl}/analyze", UnityWebRequest.kHttpVerbPOST))
            {
                req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                if (!string.IsNullOrEmpty(apiKey))
                {
                    req.SetRequestHeader("X-API-Key", apiKey);
                }
                req.timeout = timeoutSeconds;

                yield return req.SendWebRequest();

                var response = new AnalyzeResponse { raw_json = req.downloadHandler?.text };
                if (req.result != UnityWebRequest.Result.Success)
                {
                    response.error = req.error ?? "request failed";
                    response.code = "network_error";
                }
                onComplete?.Invoke(response);
            }
        }
    }
}
