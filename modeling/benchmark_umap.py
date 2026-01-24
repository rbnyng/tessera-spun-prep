"""
Benchmark: sklearn UMAP vs cuML UMAP transform speed

Tests transform speed on synthetic data similar to the inference workload.
"""

import time
import numpy as np

# Simulate a tile's worth of data
# Each tile is ~1000x1000 pixels with 128 bands, 3x3 window = 1152 features
N_SAMPLES = 500_000  # Reduced from 1M pixels for faster test
N_FEATURES = 1152    # 3x3 window * 128 bands
N_COMPONENTS = 256   # Output dimensions

print(f"Benchmark: UMAP transform speed")
print(f"  Samples: {N_SAMPLES:,}")
print(f"  Input features: {N_FEATURES}")
print(f"  Output components: {N_COMPONENTS}")
print()

# Generate random test data
np.random.seed(42)
X_test = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)

# Small training set for fitting
X_train = np.random.randn(5000, N_FEATURES).astype(np.float32)

results = {}

# --- sklearn UMAP ---
try:
    from umap import UMAP as SklearnUMAP

    print("Testing sklearn UMAP...")

    # Fit on small training set
    umap_sklearn = SklearnUMAP(n_components=N_COMPONENTS, n_neighbors=15, min_dist=0.1, random_state=42)

    t0 = time.time()
    umap_sklearn.fit(X_train)
    fit_time = time.time() - t0
    print(f"  Fit time: {fit_time:.2f}s")

    # Transform test set
    t0 = time.time()
    _ = umap_sklearn.transform(X_test)
    transform_time = time.time() - t0

    results['sklearn'] = transform_time
    print(f"  Transform time: {transform_time:.2f}s")
    print(f"  Throughput: {N_SAMPLES / transform_time:,.0f} samples/sec")
    print()

except ImportError as e:
    print(f"sklearn UMAP not available: {e}\n")

# --- cuML UMAP (GPU) ---
try:
    from cuml import UMAP as CumlUMAP
    import cupy as cp

    print("Testing cuML UMAP (GPU)...")

    # Fit on small training set
    umap_cuml = CumlUMAP(n_components=N_COMPONENTS, n_neighbors=15, min_dist=0.1, random_state=42)

    t0 = time.time()
    umap_cuml.fit(X_train)
    cp.cuda.Stream.null.synchronize()  # Ensure GPU ops complete
    fit_time = time.time() - t0
    print(f"  Fit time: {fit_time:.2f}s")

    # Transform test set
    t0 = time.time()
    _ = umap_cuml.transform(X_test)
    cp.cuda.Stream.null.synchronize()
    transform_time = time.time() - t0

    results['cuml'] = transform_time
    print(f"  Transform time: {transform_time:.2f}s")
    print(f"  Throughput: {N_SAMPLES / transform_time:,.0f} samples/sec")
    print()

except ImportError as e:
    print(f"cuML UMAP not available: {e}")
    print("Install with: conda install -c rapidsai cuml\n")

# --- Summary ---
print("=" * 50)
print("SUMMARY")
print("=" * 50)

if len(results) == 2:
    speedup = results['sklearn'] / results['cuml']
    print(f"sklearn UMAP: {results['sklearn']:.2f}s")
    print(f"cuML UMAP:    {results['cuml']:.2f}s")
    print(f"Speedup:      {speedup:.1f}x faster with GPU")
elif len(results) == 1:
    name, t = list(results.items())[0]
    print(f"{name} UMAP: {t:.2f}s")
else:
    print("No UMAP implementations available to test.")
