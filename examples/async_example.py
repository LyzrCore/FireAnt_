"""
Example demonstrating FireAnt's async capabilities.
This example shows how to use async agents for I/O-bound operations and better performance.
"""

import asyncio
import time
import random
from fireant import (
    AsyncAgent, AsyncAgentFlow, AsyncRetryPolicy, AsyncCircuitBreaker,
    run_async_flow, create_async_flow
)


class AsyncDataFetcher(AsyncAgent):
    """Async agent that fetches data from multiple sources concurrently."""
    
    def __init__(self, sources=None, delay_range=(0.1, 0.5)):
        super().__init__(
            retry_policy=AsyncRetryPolicy(max_attempts=3, delay=0.1),
            enable_monitoring=True
        )
        self.sources = sources or ["api1", "api2", "api3"]
        self.delay_range = delay_range
    
    async def fetch_from_source(self, source: str) -> dict:
        """Simulate async data fetching from a source."""
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)
        
        # Simulate occasional failures
        if random.random() < 0.2:
            raise ValueError(f"Failed to fetch from {source}")
        
        return {
            "source": source,
            "data": list(range(1, random.randint(5, 15))),
            "fetch_time": delay
        }
    
    async def execute(self, inputs):
        """Fetch data from all sources concurrently."""
        print(f"Processing {self.name}: Fetching data from {len(self.sources)} sources...")
        
        # Fetch from all sources concurrently
        tasks = [self.fetch_from_source(source) for source in self.sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Error from {self.sources[i]}: {result}")
            else:
                successful_results.append(result)
        
        if not successful_results:
            raise ValueError(f"All sources failed: {'; '.join(errors)}")
        
        # Combine all data
        all_data = []
        total_fetch_time = 0
        
        for result in successful_results:
            all_data.extend(result["data"])
            total_fetch_time += result["fetch_time"]
        
        print(f"✅ {self.name}: Fetched {len(all_data)} items in {total_fetch_time:.3f}s")
        
        return {
            "fetched_data": all_data,
            "sources_count": len(self.sources),
            "successful_sources": len(successful_results),
            "errors": errors,
            "total_fetch_time": total_fetch_time
        }


class AsyncDataProcessor(AsyncAgent):
    """Async agent that processes data in batches."""
    
    def __init__(self, batch_size=5, delay=0.05):
        super().__init__(
            retry_policy=AsyncRetryPolicy(max_attempts=2, delay=0.1),
            enable_monitoring=True
        )
        self.batch_size = batch_size
        self.delay = delay
    
    async def process_batch(self, batch: list, batch_index: int) -> list:
        """Process a single batch of data."""
        await asyncio.sleep(self.delay)
        
        # Simulate processing
        processed = [x * 2 for x in batch]
        
        # Simulate occasional batch failures
        if random.random() < 0.1:
            raise ValueError(f"Batch {batch_index} processing failed")
        
        return processed
    
    async def execute(self, inputs):
        """Process data in batches concurrently."""
        data = inputs.get("fetched_data", [])
        
        if not data:
            return {"processed_data": [], "batches_processed": 0}
        
        # Split data into batches
        batches = [
            data[i:i + self.batch_size] 
            for i in range(0, len(data), self.batch_size)
        ]
        
        print(f"Processing {self.name}: Processing {len(data)} items in {len(batches)} batches...")
        
        # Process batches concurrently
        tasks = [
            self.process_batch(batch, i) 
            for i, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        processed_data = []
        successful_batches = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️ {self.name}: Batch {i} failed: {result}")
            else:
                processed_data.extend(result)
                successful_batches += 1
        
        print(f"✅ {self.name}: Processed {len(processed_data)} items from {successful_batches}/{len(batches)} batches")
        
        return {
            "processed_data": processed_data,
            "batches_processed": successful_batches,
            "total_batches": len(batches)
        }


class AsyncDataAnalyzer(AsyncAgent):
    """Async agent that performs complex analysis."""
    
    def __init__(self):
        super().__init__(enable_monitoring=True)
    
    async def calculate_statistics(self, data: list) -> dict:
        """Calculate statistics asynchronously."""
        await asyncio.sleep(0.1)  # Simulate computation
        
        if not data:
            return {}
        
        return {
            "count": len(data),
            "sum": sum(data),
            "average": sum(data) / len(data),
            "min": min(data),
            "max": max(data)
        }
    
    async def detect_patterns(self, data: list) -> list:
        """Detect patterns in data asynchronously."""
        await asyncio.sleep(0.15)  # Simulate pattern detection
        
        patterns = []
        
        # Simple pattern detection
        if len(data) > 2:
            # Check for arithmetic progression
            diffs = [data[i+1] - data[i] for i in range(len(data)-1)]
            if len(set(diffs)) == 1:
                patterns.append("arithmetic_progression")
            
            # Check for geometric progression
            if all(x != 0 for x in data[:-1]):
                ratios = [data[i+1] / data[i] for i in range(len(data)-1)]
                if len(set(ratios)) == 1:
                    patterns.append("geometric_progression")
        
        return patterns
    
    async def execute(self, inputs):
        """Perform comprehensive data analysis."""
        data = inputs.get("processed_data", [])
        
        print(f"Processing {self.name}: Analyzing {len(data)} data points...")
        
        # Run analysis tasks concurrently
        stats_task = self.calculate_statistics(data)
        patterns_task = self.detect_patterns(data)
        
        stats, patterns = await asyncio.gather(stats_task, patterns_task)
        
        print(f"✅ {self.name}: Analysis complete. Found {len(patterns)} patterns")
        
        return {
            "statistics": stats,
            "patterns": patterns,
            "analysis_timestamp": time.time()
        }


class AsyncReportGenerator(AsyncAgent):
    """Async agent that generates reports."""
    
    def __init__(self):
        super().__init__(enable_monitoring=True)
    
    async def generate_report(self, data: dict) -> str:
        """Generate a formatted report."""
        await asyncio.sleep(0.1)  # Simulate report generation
        
        stats = data.get("statistics", {})
        patterns = data.get("patterns", [])
        
        report = []
        report.append("📊 DATA ANALYSIS REPORT")
        report.append("=" * 30)
        
        if stats:
            report.append(f"📈 Statistics:")
            report.append(f"   Count: {stats.get('count', 0)}")
            report.append(f"   Sum: {stats.get('sum', 0)}")
            report.append(f"   Average: {stats.get('average', 0):.2f}")
            report.append(f"   Min: {stats.get('min', 0)}")
            report.append(f"   Max: {stats.get('max', 0)}")
        
        if patterns:
            report.append(f"\n🔍 Patterns Detected:")
            for pattern in patterns:
                report.append(f"   - {pattern}")
        else:
            report.append(f"\n🔍 No patterns detected")
        
        report.append(f"\n⏰ Generated at: {time.strftime('%H:%M:%S')}")
        
        return "\n".join(report)
    
    async def execute(self, inputs):
        """Generate and print the final report."""
        analysis_data = inputs.get("analysis_data", {})
        
        print(f"Processing {self.name}: Generating report...")
        
        report = await self.generate_report(analysis_data)
        
        print(f"\n{report}")
        
        return {
            "report": report,
            "report_generated": True
        }


async def demonstrate_basic_async_flow():
    """Demonstrate basic async flow execution."""
    print("\n=== Basic Async Flow Demo ===")
    
    # Create async agents
    fetcher = AsyncDataFetcher(sources=["api1", "api2", "api3"])
    processor = AsyncDataProcessor(batch_size=3)
    analyzer = AsyncDataAnalyzer()
    reporter = AsyncReportGenerator()
    
    # Create flow
    flow = AsyncAgentFlow(
        start=fetcher.next(processor).next(analyzer).next(reporter),
        enable_monitoring=True
    )
    
    # Run the flow
    start_time = time.time()
    try:
        ledger = {}
        result = await flow.run(ledger)
        execution_time = time.time() - start_time
        
        print(f"\n⏱️ Total execution time: {execution_time:.3f}s")
        
        # Get monitoring summary
        summary = flow.get_monitoring_summary()
        print(f"📊 Flow success rate: {summary['flows']['success_rate']:.2%}")
        
    except Exception as e:
        print(f"❌ Flow failed: {e}")


async def demonstrate_concurrent_flows():
    """Demonstrate running multiple flows concurrently."""
    print("\n=== Concurrent Flows Demo ===")
    
    # Create multiple flows with different configurations
    flows = []
    
    for i in range(3):
        fetcher = AsyncDataFetcher(
            sources=[f"api_{i}_1", f"api_{i}_2"],
            delay_range=(0.1, 0.3)
        )
        processor = AsyncDataProcessor(batch_size=2)
        analyzer = AsyncDataAnalyzer()
        
        flow = AsyncAgentFlow(
            start=fetcher.next(processor).next(analyzer),
            enable_monitoring=True
        )
        flows.append(flow)
    
    # Run all flows concurrently
    print(f"🚀 Running {len(flows)} flows concurrently...")
    start_time = time.time()
    
    try:
        tasks = [flow.run({}) for flow in flows]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        execution_time = time.time() - start_time
        print(f"⏱️ Concurrent execution time: {execution_time:.3f}s")
        
        # Check results
        successful_flows = sum(1 for r in results if not isinstance(r, Exception))
        print(f"✅ Successful flows: {successful_flows}/{len(flows)}")
        
    except Exception as e:
        print(f"❌ Concurrent execution failed: {e}")


async def demonstrate_circuit_breaker():
    """Demonstrate async circuit breaker pattern."""
    print("\n=== Async Circuit Breaker Demo ===")
    
    class FailingAsyncAgent(AsyncAgent):
        def __init__(self, failure_rate=0.8):
            super().__init__(enable_monitoring=True)
            self.failure_rate = failure_rate
            self.call_count = 0
        
        async def execute(self, inputs):
            self.call_count += 1
            print(f"Processing {self.name} call #{self.call_count}")
            
            await asyncio.sleep(0.1)
            
            if random.random() < self.failure_rate:
                raise RuntimeError(f"Consistent failure #{self.call_count}")
            
            return {"success": True, "call": self.call_count}
    
    # Create circuit breaker with low threshold
    circuit_breaker = AsyncCircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    
    failing_agent = FailingAsyncAgent(failure_rate=0.9)
    flow = AsyncAgentFlow(
        start=failing_agent,
        circuit_breaker=circuit_breaker,
        enable_monitoring=True
    )
    
    # Try running multiple times
    for i in range(8):
        try:
            result = await flow.run({})
            print(f"✅ Attempt {i+1}: Success!")
            break
        except Exception as e:
            print(f"❌ Attempt {i+1}: {e}")
            if "Circuit breaker is OPEN" in str(e):
                print("   ⚡ Circuit breaker is OPEN - waiting...")
                await asyncio.sleep(1.5)


async def demonstrate_async_utilities():
    """Demonstrate async utility functions."""
    print("\n=== Async Utilities Demo ===")
    
    # Create agents
    fetcher = AsyncDataFetcher(sources=["util_api"])
    processor = AsyncDataProcessor()
    analyzer = AsyncDataAnalyzer()
    
    # Use utility function to create flow
    flow = create_async_flow(fetcher, processor, analyzer, enable_monitoring=True)
    
    # Use utility function to run flow
    print("🚀 Running flow with utility functions...")
    try:
        result = await run_async_flow(flow)
        print("✅ Flow completed successfully")
    except Exception as e:
        print(f"❌ Flow failed: {e}")


async def demonstrate_mixed_sync_async():
    """Demonstrate mixing sync and async agents."""
    print("\n=== Mixed Sync/Async Demo ===")
    
    # Import sync agent from core
    from fireant import Agent
    
    class SyncDataValidator(Agent):
        def execute(self, inputs):
            data = inputs.get("processed_data", [])
            print(f"🔍 {self.name}: Validating {len(data)} items (sync)")
            
            # Simple validation
            valid_data = [x for x in data if x > 0]
            
            return {
                "validated_data": valid_data,
                "invalid_count": len(data) - len(valid_data)
            }
    
    # Create mixed flow
    async_fetcher = AsyncDataFetcher(sources=["mixed_api"])
    async_processor = AsyncDataProcessor()
    sync_validator = SyncDataValidator()
    async_analyzer = AsyncDataAnalyzer()
    
    flow = AsyncAgentFlow(
        start=async_fetcher.next(async_processor).next(sync_validator).next(async_analyzer),
        enable_monitoring=True
    )
    
    print("Running mixed sync/async flow...")
    try:
        result = await flow.run({})
        print("✅ Mixed flow completed successfully")
    except Exception as e:
        print(f"❌ Mixed flow failed: {e}")


async def main():
    """Run all async demonstrations."""
    print("FireAnt Async Examples")
    print("=" * 50)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run demonstrations
    await demonstrate_basic_async_flow()
    await demonstrate_concurrent_flows()
    await demonstrate_circuit_breaker()
    await demonstrate_async_utilities()
    await demonstrate_mixed_sync_async()
    
    print("\n✨ Async demo completed!")


if __name__ == "__main__":
    asyncio.run(main())