# ASAP Performance Optimization Guide

This document provides guidance on improving the performance of ASAP, particularly when using it with web-based tools or interactive applications.

## Recent Optimizations (v0.1.1)

### Core Algorithm Improvements

1. **Region Creation (`make_regions_order_revamp`)**
   - Replaced inefficient `list.remove()` operations with index-based list manipulation
   - Used `np.searchsorted()` for fast binary search instead of boolean indexing
   - Pre-computed wavelength bounds to avoid repeated comparisons
   - Replaced temporary array creation with direct slicing operations
   - **Expected speedup**: 2-5x for typical line lists

2. **Bounds Merging (`merge`)**
   - Vectorized empty bound filtering
   - Simplified control flow logic
   - Early returns for edge cases
   - **Expected speedup**: 1.5-3x

3. **Bounds Width Adjustment (`increase_width`)**
   - Fully vectorized using numpy operations
   - Eliminated Python loops
   - **Expected speedup**: 5-10x for large lists

4. **Region Assembly (`make_regions_2d_orders`)**
   - Optimized array flattening with list comprehension and zip
   - Pre-filtering of invalid regions
   - Vectorized operations where possible
   - **Expected speedup**: 1.5-2x

### Memory Efficiency

- Reduced temporary array allocations in hot paths
- Used C-contiguous arrays for better cache locality
- Pre-allocated output arrays when size is known

## Best Practices for Web Tools

### 1. Caching Strategy

For web-based curator tools, implement caching at multiple levels:

```python
from asap.performance_utils import cache_result

# Cache expensive region computations
@cache_result(maxsize=256)
def compute_regions(region_file, wvl_data, flux_data):
    # ... region computation ...
    return regions
```

### 2. Lazy Loading

Load data only when needed:

```python
class LazySpectralData:
    def __init__(self, filename):
        self.filename = filename
        self._data = None
    
    @property
    def data(self):
        if self._data is None:
            self._data = load_data(self.filename)
        return self._data
```

### 3. Batch Processing

Process multiple regions or spectra in batches:

```python
# Instead of processing one at a time
for region in regions:
    result = process_region(region)

# Process in batches
batch_size = 10
for i in range(0, len(regions), batch_size):
    batch = regions[i:i+batch_size]
    results = process_regions_batch(batch)
```

### 4. Grid Data Preloading

For interactive tools, preload and cache grid data:

```python
# Preload grid on startup
grid_cache = {}

def load_grid_cached(pathtogrid):
    if pathtogrid not in grid_cache:
        grid_cache[pathtogrid] = load_grid(pathtogrid)
    return grid_cache[pathtogrid]
```

## Configuration Options

### NumPy Performance

Set these environment variables before importing NumPy:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

### Numba JIT Compilation

The first call to JIT-compiled functions will be slow due to compilation.
Warm up the cache:

```python
import asap.line_selection_tools as lst
import numpy as np

# Warm up JIT cache
dummy_wvl = np.linspace(1000, 2000, 1000)
dummy_flux = np.ones_like(dummy_wvl)
dummy_lines = np.array([1100, 1500, 1900])
_ = lst.select_lines(dummy_lines, dummy_wvl, dummy_flux)
```

## Profiling Your Code

To identify bottlenecks in your specific use case:

```python
import cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()

# Your ASAP code here
# ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Common Performance Issues

### Issue: Slow region creation
**Solution**: Use the optimized `make_regions_order_revamp` function (default in v0.1.1+)

### Issue: Web tool hangs on large datasets
**Solutions**:
- Implement progress indicators
- Use asynchronous processing
- Break large operations into smaller chunks
- Cache intermediate results

### Issue: High memory usage
**Solutions**:
- Process data in streaming fashion rather than loading all at once
- Use memory-mapped arrays for large grid data
- Clear caches periodically

### Issue: Repeated computations
**Solutions**:
- Use the `@cache_result` decorator from `performance_utils`
- Implement session-based caching in web tools
- Precompute and store expensive results

## Web Tool Specific Tips

### For Dash/Plotly Applications

1. **Use callback caching**:
```python
from dash import callback_context
from functools import lru_cache

@app.callback(...)
@lru_cache(maxsize=128)
def update_plot(region_id):
    # ...
```

2. **Implement loading states**:
```python
dcc.Loading(
    id="loading",
    type="circle",
    children=html.Div(id="output")
)
```

3. **Use clientside callbacks** for simple updates:
```python
app.clientside_callback(
    """
    function(value) {
        return value + ' (processed client-side)';
    }
    """,
    Output('output', 'children'),
    Input('input', 'value')
)
```

4. **Optimize figure updates**:
- Use `Patch()` for incremental updates
- Update only changed traces instead of recreating entire figures

## Monitoring Performance

Track key metrics in your web tool:

```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
    
    def time_operation(self, name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(elapsed)
                
                return result
            return wrapper
        return decorator
    
    def report(self):
        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            print(f"{name}: avg={avg:.3f}s, calls={len(times)}")

monitor = PerformanceMonitor()

@monitor.time_operation("region_creation")
def create_regions(...):
    # ...
```

## Support

For performance issues or questions:
- Check the issue tracker on GitHub
- Profile your specific use case
- Consider contributing optimizations back to the project
