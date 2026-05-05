using System;
using UnityEngine;

namespace DeepFaceAge
{
    /// <summary>
    /// ScriptableObject holding remote endpoint and credentials for the
    /// AgeEstimatorClient. Place an instance under Resources/ to load defaults.
    /// </summary>
    [CreateAssetMenu(menuName = "DeepFaceAge/Settings", fileName = "AgeEstimatorSettings")]
    public class AgeEstimatorSettings : ScriptableObject
    {
        [Tooltip("Base URL of the DeepFace server, e.g. http://localhost:5000")]
        public string baseUrl = "http://localhost:5000";

        [Tooltip("Optional API key sent as X-API-Key header.")]
        public string apiKey = string.Empty;

        [Tooltip("Optional bearer token sent as Authorization: Bearer <token>.")]
        public string bearerToken = string.Empty;

        [Tooltip("Per-request HTTP timeout (seconds).")]
        public int requestTimeoutSeconds = 30;

        [Tooltip("Number of retries on transient HTTP failures.")]
        public int maxRetries = 2;

        [Tooltip("Initial backoff in milliseconds between retries.")]
        public int initialBackoffMs = 250;

        public bool HasApiKey => !string.IsNullOrEmpty(apiKey);
        public bool HasBearer => !string.IsNullOrEmpty(bearerToken);

        public Uri AnalyzeUri => new Uri($"{baseUrl.TrimEnd('/')}/analyze");
        public Uri BatchUri => new Uri($"{baseUrl.TrimEnd('/')}/analyze/batch");
        public Uri JobsUri => new Uri($"{baseUrl.TrimEnd('/')}/jobs");
    }
}
