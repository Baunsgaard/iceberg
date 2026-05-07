/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.arrow.vectorized;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.FileContent;
import org.apache.iceberg.FileFormat;
import org.apache.iceberg.Files;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.data.BaseDeleteLoader;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.deletes.PositionDelete;
import org.apache.iceberg.deletes.PositionDeleteWriter;
import org.apache.iceberg.formats.FormatModelRegistry;
import org.apache.iceberg.formats.PositionDeleteIndexReader;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.relocated.com.google.common.base.Preconditions;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.mockito.Mockito;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Timeout;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * End-to-end benchmark that verifies the {@link FormatModelRegistry} integration of {@link
 * VectorizedPositionDeleteReader} via {@link BaseDeleteLoader}.
 *
 * <p>The benchmark exercises {@link BaseDeleteLoader#loadPositionDeletes} in two registry states:
 *
 * <ul>
 *   <li>{@code fast} -- {@link ArrowFormatModels} has registered Arrow's reader for parquet, so
 *       {@code BaseDeleteLoader} dispatches through {@link VectorizedPositionDeleteReader#read}.
 *   <li>{@code slow} -- the parquet reader has been removed from the registry, so the loader falls
 *       back to the per-row record reader plus {@link
 *       org.apache.iceberg.deletes.Deletes#toPositionIndex(CharSequence,
 *       org.apache.iceberg.io.CloseableIterable, DeleteFile)}.
 * </ul>
 *
 * <p>Both modes run against the same input file, so any difference between them is attributable to
 * the registry-driven dispatch path. A measurable speed-up in the {@code fast} mode confirms the
 * integration works end-to-end -- the loader is actually picking up the registered reader rather
 * than silently falling through to the generic path.
 *
 * <p>Reflection is used to mutate {@link FormatModelRegistry#positionDeleteIndexReaders}, which is
 * marked {@code @VisibleForTesting} and intentionally has no public {@code unregister} method.
 *
 * <p>To run: <code>
 *   ./gradlew :iceberg-arrow:jmh
 *       -PjmhIncludeRegex=BaseDeleteLoaderBenchmark
 *       -PjmhOutputPath=benchmark/base-delete-loader-benchmark.txt
 * </code>
 */
@Fork(1)
@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 3)
@Measurement(iterations = 5, time = 5)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Timeout(time = 5, timeUnit = TimeUnit.MINUTES)
public class BaseDeleteLoaderBenchmark {

  private static final String DATA_LOCATION = "s3://bucket/path/to/data-file.parquet";
  private static final int NUM_POSITIONS = 1_000_000;
  private static final long SPARSE_STRIDE = 100L;

  @Param({"dense", "sparse"})
  private String distribution;

  @Param({"fast", "slow"})
  private String registryMode;

  private File deleteFile;
  private DeleteFile deleteFileMetadata;
  private BaseDeleteLoader loader;
  private PositionDeleteIndexReader cachedArrowReader;

  @Setup(Level.Trial)
  public void setupTrial() throws IOException {
    deleteFile =
        java.nio.file.Files.createTempFile("base-delete-loader-bench", ".parquet").toFile();
    deleteFile.delete();

    long[] positions = "dense".equals(distribution) ? densePositions() : sparsePositions();
    writeDeleteFile(deleteFile, positions);

    deleteFileMetadata = mockDeleteFile(deleteFile, positions.length);
    // Use a caching loader so loadPositionDeletes() takes the multi-file readAll() path,
    // which is where the vectorized reader provides the largest speed-up. Without a
    // cache, the loader filters by data-file path inline and the fast/slow paths are
    // dominated by per-row path comparisons (see PositionDeleteReaderBenchmark's
    // 'filtered' vs 'noFilter' results).
    loader = new CachingDeleteLoader(unused -> Files.localInput(deleteFile));

    // Capture the parquet reader registered by ArrowFormatModels' static init so the
    // 'fast' mode can be reset to a known good state regardless of test ordering.
    cachedArrowReader = readersMap().get(FileFormat.PARQUET);
    Preconditions.checkState(
        cachedArrowReader != null,
        "ArrowFormatModels should have registered a parquet position delete index reader");
  }

  @Setup(Level.Iteration)
  public void setupIteration() {
    Map<FileFormat, PositionDeleteIndexReader> readers = readersMap();
    if ("fast".equals(registryMode)) {
      readers.put(FileFormat.PARQUET, cachedArrowReader);
    } else {
      readers.remove(FileFormat.PARQUET);
    }
  }

  @Setup(Level.Invocation)
  public void setupInvocation() {
    // Drop the cached entry before each measured invocation so the benchmark exercises
    // the readAll() path end-to-end rather than returning the previous cached index.
    if (loader instanceof CachingDeleteLoader) {
      ((CachingDeleteLoader) loader).clearCache();
    }
  }

  @TearDown(Level.Trial)
  public void teardownTrial() {
    // Restore the registry so other benchmarks running in the same JVM see the original state.
    readersMap().put(FileFormat.PARQUET, cachedArrowReader);
    if (deleteFile != null) {
      deleteFile.delete();
    }
  }

  @Benchmark
  @Threads(1)
  public void loadPositionDeletes(Blackhole blackhole) {
    blackhole.consume(
        loader.loadPositionDeletes(ImmutableList.of(deleteFileMetadata), DATA_LOCATION));
  }

  private static DeleteFile mockDeleteFile(File file, long recordCount) {
    DeleteFile metadata = Mockito.mock(DeleteFile.class);
    Mockito.when(metadata.format()).thenReturn(FileFormat.PARQUET);
    Mockito.when(metadata.content()).thenReturn(FileContent.POSITION_DELETES);
    Mockito.when(metadata.location()).thenReturn(file.getAbsolutePath());
    Mockito.when(metadata.recordCount()).thenReturn(recordCount);
    return metadata;
  }

  @SuppressWarnings("unchecked")
  private static Map<FileFormat, PositionDeleteIndexReader> readersMap() {
    try {
      Method method = FormatModelRegistry.class.getDeclaredMethod("positionDeleteIndexReaders");
      method.setAccessible(true);
      return (Map<FileFormat, PositionDeleteIndexReader>) method.invoke(null);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(
          "Cannot access FormatModelRegistry.positionDeleteIndexReaders()", e);
    }
  }

  private static void writeDeleteFile(File file, long[] positions) throws IOException {
    OutputFile out = Files.localOutput(file);
    PositionDelete<Void> pd = PositionDelete.create();
    try (PositionDeleteWriter<Void> writer =
        Parquet.writeDeletes(out)
            .createWriterFunc(GenericParquetWriter::create)
            .overwrite()
            .withSpec(PartitionSpec.unpartitioned())
            .buildPositionWriter()) {
      for (long position : positions) {
        pd.set(DATA_LOCATION, position, null);
        writer.write(pd);
      }
    }
  }

  private static long[] densePositions() {
    long[] positions = new long[NUM_POSITIONS];
    for (int i = 0; i < NUM_POSITIONS; i++) {
      positions[i] = i;
    }
    return positions;
  }

  private static long[] sparsePositions() {
    long[] positions = new long[NUM_POSITIONS];
    for (int i = 0; i < NUM_POSITIONS; i++) {
      positions[i] = i * SPARSE_STRIDE;
    }
    return positions;
  }

  /**
   * Minimal caching loader so {@link BaseDeleteLoader#loadPositionDeletes} routes through the
   * multi-file {@code readAll()} path. The cache is cleared between benchmark iterations so each
   * measured invocation actually re-reads the delete file.
   */
  private static final class CachingDeleteLoader extends BaseDeleteLoader {
    private final Map<String, Object> cache = new ConcurrentHashMap<>();

    CachingDeleteLoader(
        java.util.function.Function<DeleteFile, org.apache.iceberg.io.InputFile> loadInputFile) {
      super(loadInputFile);
    }

    @Override
    protected boolean canCache(long size) {
      return true;
    }

    @Override
    @SuppressWarnings("unchecked")
    protected <V> V getOrLoad(String key, Supplier<V> valueSupplier, long valueSize) {
      return (V) cache.computeIfAbsent(key, unused -> valueSupplier.get());
    }

    void clearCache() {
      cache.clear();
    }
  }
}
