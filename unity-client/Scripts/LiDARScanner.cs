using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class LiDARScanner : MonoBehaviour
{
    public void SendLiDARHit(Vector3 hitPoint)
    {
        StartCoroutine(PostLiDARData(hitPoint));
    }

    private IEnumerator PostLiDARData(Vector3 hitPoint)
    {
        string url = "http://localhost:8000/api/lidar";
        string jsonPayload = JsonUtility.ToJson(new { x = hitPoint.x, y = hitPoint.y, z = hitPoint.z });

        using (UnityWebRequest request = UnityWebRequest.Post(url, jsonPayload))
        {
            request.SetRequestHeader("Content-Type", "application/json");
            request.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(jsonPayload));
            request.downloadHandler = new DownloadHandlerBuffer();

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
                Debug.Log("LiDAR hit sent successfully.");
            else
                Debug.LogError($"Failed to send LiDAR hit: {request.error}");
        }
    }
}