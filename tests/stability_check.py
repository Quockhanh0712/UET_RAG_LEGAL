"""
Load/Stability Test - Concurrent Request Simulation.
Tests API under load to detect memory leaks and performance issues.
"""
import sys
import time
import asyncio
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


@dataclass
class RequestResult:
    """Result of a single request."""
    success: bool
    latency_ms: float
    status_code: int
    error: Optional[str] = None


@dataclass
class LoadTestResult:
    """Overall load test results."""
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    total_time_s: float
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "error_rate": round(self.error_rate * 100, 2),
            "total_time_s": round(self.total_time_s, 2),
            "errors": self.errors[:10]  # Limit to first 10 errors
        }


class LoadTester:
    """Load tester for API endpoints."""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 60.0
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.results: List[RequestResult] = []
    
    async def send_request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        method: str = "POST",
        json_data: Optional[dict] = None
    ) -> RequestResult:
        """Send a single request and record result."""
        url = f"{self.base_url}{endpoint}"
        start = time.time()
        
        try:
            if method == "POST":
                response = await client.post(url, json=json_data, timeout=self.timeout)
            else:
                response = await client.get(url, timeout=self.timeout)
            
            latency = (time.time() - start) * 1000
            
            return RequestResult(
                success=response.status_code == 200,
                latency_ms=latency,
                status_code=response.status_code,
                error=None if response.status_code == 200 else response.text[:200]
            )
            
        except httpx.TimeoutException:
            latency = (time.time() - start) * 1000
            return RequestResult(
                success=False,
                latency_ms=latency,
                status_code=0,
                error="Timeout"
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return RequestResult(
                success=False,
                latency_ms=latency,
                status_code=0,
                error=str(e)[:200]
            )
    
    async def run_load_test(
        self,
        endpoint: str,
        num_requests: int = 20,
        concurrency: int = 5,
        method: str = "POST",
        json_data: Optional[dict] = None
    ) -> LoadTestResult:
        """
        Run load test with concurrent requests.
        
        Args:
            endpoint: API endpoint to test
            num_requests: Total number of requests
            concurrency: Max concurrent requests
            method: HTTP method
            json_data: JSON payload for POST
            
        Returns:
            LoadTestResult with statistics
        """
        print(f"\n🔄 Running load test: {endpoint}")
        print(f"   Requests: {num_requests}, Concurrency: {concurrency}")
        
        self.results = []
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_request():
                async with semaphore:
                    result = await self.send_request(client, endpoint, method, json_data)
                    self.results.append(result)
                    
                    # Progress indicator
                    done = len(self.results)
                    if done % 5 == 0 or done == num_requests:
                        print(f"   Progress: {done}/{num_requests}")
                    
                    return result
            
            tasks = [bounded_request() for _ in range(num_requests)]
            await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        latencies = [r.latency_ms for r in self.results]
        successful = sum(1 for r in self.results if r.success)
        failed = num_requests - successful
        errors = [r.error for r in self.results if r.error]
        
        sorted_latencies = sorted(latencies)
        p50_idx = int(len(sorted_latencies) * 0.5)
        p95_idx = int(len(sorted_latencies) * 0.95)
        
        result = LoadTestResult(
            total_requests=num_requests,
            successful=successful,
            failed=failed,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p50_latency_ms=sorted_latencies[p50_idx] if sorted_latencies else 0,
            p95_latency_ms=sorted_latencies[p95_idx] if sorted_latencies else 0,
            error_rate=failed / num_requests if num_requests > 0 else 0,
            total_time_s=total_time,
            errors=list(set(errors))  # Unique errors
        )
        
        # Print summary
        print(f"\n📊 Results:")
        print(f"   ✓ Success: {successful}/{num_requests} ({100 - result.error_rate * 100:.1f}%)")
        print(f"   ✗ Failed: {failed}")
        print(f"   ⏱ Avg Latency: {result.avg_latency_ms:.0f}ms")
        print(f"   ⏱ P95 Latency: {result.p95_latency_ms:.0f}ms")
        
        return result


async def run_stability_check(base_url: str = "http://127.0.0.1:8080") -> dict:
    """
    Run full stability check suite.
    
    Tests:
    1. Status endpoint (light)
    2. Search endpoint (medium)
    3. Chat endpoint (heavy)
    """
    tester = LoadTester(base_url=base_url)
    results = {}
    
    print("=" * 60)
    print("🧪 BACKEND STABILITY CHECK")
    print(f"   Target: {base_url}")
    print(f"   Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Test 1: Status endpoint (light load)
    results["status"] = await tester.run_load_test(
        endpoint="/api/status/quick",
        num_requests=20,
        concurrency=10,
        method="GET"
    )
    
    # Test 2: Search endpoint (medium load)
    results["search"] = await tester.run_load_test(
        endpoint="/api/search",
        num_requests=10,
        concurrency=3,
        method="POST",
        json_data={
            "query": "Tội tham nhũng bị phạt bao nhiêu năm?",
            "top_k": 5,
            "search_mode": "legal",
            "reranker_enabled": False
        }
    )
    
    # Test 3: Chat endpoint (heavy load - includes LLM)
    results["chat"] = await tester.run_load_test(
        endpoint="/api/chat",
        num_requests=5,
        concurrency=2,
        method="POST",
        json_data={
            "message": "Tội tham nhũng bị phạt thế nào?",
            "user_id": "load_test_user",
            "search_mode": "legal",
            "reranker_enabled": False
        }
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results.items():
        status = "✅ PASS" if result.error_rate < 0.1 else "❌ FAIL"
        if result.error_rate >= 0.1:
            all_passed = False
        print(f"   {name}: {status} (Error rate: {result.error_rate * 100:.1f}%)")
    
    overall_status = "✅ STABLE" if all_passed else "❌ UNSTABLE"
    print(f"\n   Overall: {overall_status}")
    
    return {
        "status": "STABLE" if all_passed else "UNSTABLE",
        "timestamp": datetime.now().isoformat(),
        "tests": {name: r.to_dict() for name, r in results.items()}
    }


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Backend Stability Check")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()
    
    result = asyncio.run(run_stability_check(args.url))
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
