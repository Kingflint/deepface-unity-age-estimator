using System;
using System.Collections.Generic;

namespace DeepFaceAge.Models
{
    [Serializable]
    public class FaceRegion
    {
        public int x;
        public int y;
        public int w;
        public int h;
    }

    [Serializable]
    public class AnalysisResult
    {
        public float? age;
        public float? ageConfidence;
        public string dominantEmotion;
        public Dictionary<string, float> emotionScores = new Dictionary<string, float>();
        public string dominantGender;
        public Dictionary<string, float> genderScores = new Dictionary<string, float>();
        public FaceRegion region = new FaceRegion();
        public string backend;

        public bool HasFace => region != null && region.w > 0 && region.h > 0;

        public override string ToString()
        {
            return $"AnalysisResult(age={age:F1}, emotion={dominantEmotion}, gender={dominantGender})";
        }
    }
}
