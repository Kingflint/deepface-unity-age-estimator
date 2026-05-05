using System;

namespace DeepFaceAge.Models
{
    public enum JobState
    {
        Pending,
        Running,
        Succeeded,
        Failed,
        Cancelled,
    }

    [Serializable]
    public class JobStatus
    {
        public string id;
        public string kind;
        public string status;
        public string submittedAt;
        public string startedAt;
        public string finishedAt;
        public AnalysisResult result;
        public string error;

        public JobState State
        {
            get
            {
                switch (status)
                {
                    case "pending":
                        return JobState.Pending;
                    case "running":
                        return JobState.Running;
                    case "succeeded":
                        return JobState.Succeeded;
                    case "failed":
                        return JobState.Failed;
                    case "cancelled":
                        return JobState.Cancelled;
                    default:
                        return JobState.Pending;
                }
            }
        }

        public bool IsTerminal
        {
            get
            {
                var s = State;
                return s == JobState.Succeeded || s == JobState.Failed || s == JobState.Cancelled;
            }
        }
    }
}
