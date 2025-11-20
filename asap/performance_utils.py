"""
Performance utilities for ASAP

This module provides caching and optimization utilities to improve
the performance of ASAP, particularly for web-based tools that make
repeated calls to the analysis functions.
"""

import functools
import hashlib
import pickle
from typing import Any, Callable
import numpy as np


class LRUCache:
    """
    Simple Least Recently Used cache implementation for caching
    expensive function results.
    """
    
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()


def make_hashable(obj: Any) -> bytes:
    """
    Convert an object to a hashable representation.
    Handles numpy arrays, lists, tuples, and basic types.
    """
    if isinstance(obj, np.ndarray):
        return hashlib.md5(obj.tobytes()).digest()
    elif isinstance(obj, (list, tuple)):
        return hashlib.md5(str(obj).encode()).digest()
    elif isinstance(obj, dict):
        items = sorted(obj.items())
        return hashlib.md5(str(items).encode()).digest()
    else:
        return hashlib.md5(str(obj).encode()).digest()


def cache_result(maxsize=128):
    """
    Decorator to cache function results.
    Useful for expensive computations that are called repeatedly
    with the same parameters.
    
    Args:
        maxsize: Maximum number of cached results to keep
    
    Example:
        @cache_result(maxsize=256)
        def expensive_function(wvl, flux):
            # ... expensive computation ...
            return result
    """
    cache = LRUCache(maxsize)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            key_parts = []
            for arg in args:
                key_parts.append(make_hashable(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(make_hashable(k))
                key_parts.append(make_hashable(v))
            
            cache_key = b''.join(key_parts)
            
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = lambda: {
            'size': len(cache.cache),
            'maxsize': cache.maxsize
        }
        
        return wrapper
    
    return decorator


def optimize_array_operations(func: Callable) -> Callable:
    """
    Decorator that ensures numpy arrays are C-contiguous for better performance.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert args to C-contiguous arrays if they're numpy arrays
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray) and not arg.flags['C_CONTIGUOUS']:
                new_args.append(np.ascontiguousarray(arg))
            else:
                new_args.append(arg)
        
        # Same for kwargs
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray) and not v.flags['C_CONTIGUOUS']:
                new_kwargs[k] = np.ascontiguousarray(v)
            else:
                new_kwargs[k] = v
        
        return func(*new_args, **new_kwargs)
    
    return wrapper


def batch_operation(batch_size=10):
    """
    Decorator to process operations in batches for better cache locality
    and reduced overhead.
    
    Args:
        batch_size: Number of items to process in each batch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(items, *args, **kwargs):
            if not isinstance(items, (list, tuple, np.ndarray)):
                return func(items, *args, **kwargs)
            
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                batch_results = func(batch, *args, **kwargs)
                results.extend(batch_results)
            
            return results
        
        return wrapper
    
    return decorator
