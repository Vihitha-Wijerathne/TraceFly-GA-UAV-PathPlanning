using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class BandwidthManager : MonoBehaviour
{
    public static string bandwidthMode = "high";

    void Start()
    {
        StartCoroutine(CheckBandwidthMode());
    }

    private IEnumerator CheckBandwidthMode()
    {
        string url = "http://localhost:8000/api/bandwidth";

        while (true)
        {
            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    BandwidthResponse response = JsonUtility.FromJson<BandwidthResponse>(request.downloadHandler.text);
                    bandwidthMode = response.mode;
                    Debug.Log($"Bandwidth mode: {bandwidthMode}");
                }
                else
                {
                    Debug.LogError($"Failed to fetch bandwidth mode: {request.error}");
                }
            }

            yield return new WaitForSeconds(5);
        }
    }

    [System.Serializable]
    private class BandwidthResponse
    {
        public string mode;
    }
}