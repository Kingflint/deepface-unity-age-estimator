using System;
using System.Collections.Generic;

namespace DeepFaceAge.Models
{
    [Serializable]
    public class BatchItemResult
    {
        public int index;
        public string status; // "ok" | "error"
        public AnalysisResult result;
        public string error;
        public string code;
    }

    [Serializable]
    public class BatchResult
    {
        public List<BatchItemResult> items = new List<BatchItemResult>();
        public int successCount;
        public int failureCount;
        public float durationMs;
    }
}
