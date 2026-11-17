
import json
import time
import math
import statistics
import urllib.request

API_URL = "http://127.0.0.1:8000/predict"

WARMUP_REQUESTS = 10
MEASURED_REQUESTS = 100


def send_request(visitor_id):
    payload = {
        "visitor_id": visitor_id,
        "top_k": 10
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    end = time.perf_counter()

    assert response.status == 200
    assert result["total_recommendations"] <= 10

    return (end - start) * 1000


def benchmark_user(visitor_id, user_type):

    print(f"\nBenchmarking {user_type} user...")

    # Warm-up requests
    for _ in range(WARMUP_REQUESTS):
        send_request(visitor_id)

    # Measured requests
    latencies = []

    for _ in range(MEASURED_REQUESTS):
        latency = send_request(visitor_id)
        latencies.append(latency)

    sorted_latencies = sorted(latencies)

    p95_index = math.ceil(0.95 * len(sorted_latencies)) - 1
    p95 = sorted_latencies[p95_index]

    print(f"Requests measured : {len(latencies)}")
    print(f"Average latency   : {statistics.mean(latencies):.2f} ms")
    print(f"Median latency    : {statistics.median(latencies):.2f} ms")
    print(f"P95 latency       : {p95:.2f} ms")
    print(f"Maximum latency   : {max(latencies):.2f} ms")


if __name__ == "__main__":

    print("E-Commerce Recommendation API Benchmark")
    print("----------------------------------------")

    benchmark_user(7, "Known")
    benchmark_user(-1, "Cold-start")

    print("\nBenchmark completed successfully.")
