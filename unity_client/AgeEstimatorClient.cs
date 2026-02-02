using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace DeepFaceAge
{
    public class AgeEstimatorClient : MonoBehaviour
    {
        public string baseUrl = "http://localhost:5000";

        public IEnumerator Analyze(Texture2D texture, Action<string> onComplete)
        {
            byte[] png = texture.EncodeToPNG();
            string body = "{\"image\":\"" + Convert.ToBase64String(png) + "\"}";
            using (var req = UnityWebRequest.Post($"{baseUrl}/analyze", body))
            {
                yield return req.SendWebRequest();
                onComplete?.Invoke(req.downloadHandler.text);
            }
        }
    }
}