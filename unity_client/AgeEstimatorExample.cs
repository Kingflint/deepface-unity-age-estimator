using UnityEngine;

namespace DeepFaceAge
{
    /// <summary>
    /// Example showing how to grab a frame from a webcam and POST it.
    /// Attach this script alongside <see cref="AgeEstimatorClient"/>.
    /// </summary>
    [RequireComponent(typeof(AgeEstimatorClient))]
    public class AgeEstimatorExample : MonoBehaviour
    {
        public int requestEveryNFrames = 60;

        private AgeEstimatorClient _client;
        private WebCamTexture _webcam;
        private Texture2D _snapshot;
        private int _frame;

        private void Start()
        {
            _client = GetComponent<AgeEstimatorClient>();
            _webcam = new WebCamTexture();
            _webcam.Play();
        }

        private void Update()
        {
            if (_webcam == null || !_webcam.isPlaying) return;
            _frame++;
            if (_frame % Mathf.Max(1, requestEveryNFrames) != 0) return;

            if (_snapshot == null || _snapshot.width != _webcam.width)
            {
                _snapshot = new Texture2D(_webcam.width, _webcam.height, TextureFormat.RGB24, false);
            }
            _snapshot.SetPixels(_webcam.GetPixels());
            _snapshot.Apply();

            StartCoroutine(_client.Analyze(_snapshot, OnAnalyzeResult));
        }

        private void OnAnalyzeResult(AnalyzeResponse response)
        {
            if (!string.IsNullOrEmpty(response.error))
            {
                Debug.LogError($"DeepFace error: {response.error} ({response.code})");
                return;
            }
            Debug.Log($"DeepFace OK: {response.raw_json}");
        }

        private void OnDestroy()
        {
            if (_webcam != null && _webcam.isPlaying)
            {
                _webcam.Stop();
            }
        }
    }
}
