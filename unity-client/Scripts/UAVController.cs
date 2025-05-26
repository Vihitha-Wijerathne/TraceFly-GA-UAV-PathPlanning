using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;

public class UAVController : MonoBehaviour
{
    public GameObject waypointPrefab;
    private List<Vector3> waypoints = new List<Vector3>();

    void Start()
    {
        StartCoroutine(FetchWaypoints());
    }

    private IEnumerator FetchWaypoints()
    {
        string url = "http://localhost:8000/api/waypoints";

        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                WaypointResponse response = JsonUtility.FromJson<WaypointResponse>(request.downloadHandler.text);
                foreach (var wp in response.waypoints)
                {
                    Vector3 waypoint = new Vector3(wp.x, wp.y, wp.z);
                    waypoints.Add(waypoint);
                    Instantiate(waypointPrefab, waypoint, Quaternion.identity);
                }
                Debug.Log("Waypoints fetched and assigned.");
            }
            else
            {
                Debug.LogError($"Failed to fetch waypoints: {request.error}");
            }
        }
    }

    [System.Serializable]
    private class WaypointResponse
    {
        public List<Waypoint> waypoints;
    }

    [System.Serializable]
    private class Waypoint
    {
        public float x, y, z;
    }
}