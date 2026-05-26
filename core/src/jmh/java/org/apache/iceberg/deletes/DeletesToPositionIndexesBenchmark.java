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
package org.apache.iceberg.deletes;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import org.apache.iceberg.Accessor;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.MetadataColumns;
import org.apache.iceberg.Schema;
import org.apache.iceberg.StructLike;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.util.CharSequenceMap;
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
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Timeout;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Microbenchmark for {@link Deletes#toPositionIndexes(CloseableIterable)}. Each invocation streams
 * a pre-built {@code List<StructLike>} of {@value #TOTAL_POSITIONS} records through the production
 * coalesced path and through a naive baseline that mirrors the pre-PR per-position implementation.
 *
 * <p>The benchmark exercises the {@code CharSequenceMap<PositionDeleteIndex>} return path used by
 * {@code BaseDeleteLoader} when the {@code FormatModelRegistry} does not register a faster
 * format-specific reader -- i.e. Flink reads, and Avro/ORC delete files in any engine. The Arrow
 * Parquet fast path bypasses this method entirely and is covered by {@code
 * BaseDeleteLoaderBenchmark} in the {@code iceberg-arrow} module.
 *
 * <p>Parameters:
 *
 * <ul>
 *   <li>{@code distribution} -- shape of positions within each path:
 *       <ul>
 *         <li>{@code FULL} -- one contiguous run.
 *         <li>{@code MEDIUM} -- ~64-position runs separated by single-position gaps.
 *         <li>{@code SHORT} -- ~4-position runs separated by single-position gaps.
 *         <li>{@code SPARSE_50} -- 50% of adjacent pairs cross a gap.
 *         <li>{@code NONE} -- every other position (no consecutive pairs at all).
 *       </ul>
 *   <li>{@code paths} -- number of distinct data-file paths the deletes reference. Records are
 *       grouped by adjacent path, modelling spec-sorted delete files.
 * </ul>
 *
 * <p>Strategies compared:
 *
 * <ul>
 *   <li>{@link #coalescedToPositionIndexes} -- the production path, which groups by adjacent {@code
 *       file_path} via a peeking iterator and feeds each run into {@link
 *       PositionDeleteRangeConsumer#forEach(Iterable, PositionDeleteIndex)}.
 *   <li>{@link #naivePerPositionBaseline} -- inlined copy of the pre-PR implementation: a single
 *       per-row loop calling {@code index.delete(position)} with no coalescing. Kept as a fixed
 *       counterfactual so regressions against the historical baseline stay visible.
 * </ul>
 *
 * <p>To run: <code>
 *   ./gradlew :iceberg-core:jmh
 *       -PjmhIncludeRegex=DeletesToPositionIndexesBenchmark
 *       -PjmhOutputPath=benchmark/deletes-to-position-indexes-benchmark.txt
 * </code>
 */
@Fork(1)
@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Timeout(time = 5, timeUnit = TimeUnit.MINUTES)
public class DeletesToPositionIndexesBenchmark {

  private static final int TOTAL_POSITIONS = 1_000_000;

  // Replicates the package-private accessors in Deletes.java so the naive baseline reads the
  // record fields through the same Accessor<StructLike> dispatch the production path used to.
  private static final Schema POSITION_DELETE_SCHEMA =
      new Schema(MetadataColumns.DELETE_FILE_PATH, MetadataColumns.DELETE_FILE_POS);
  private static final Accessor<StructLike> FILENAME_ACCESSOR =
      POSITION_DELETE_SCHEMA.accessorForField(MetadataColumns.DELETE_FILE_PATH.fieldId());
  private static final Accessor<StructLike> POSITION_ACCESSOR =
      POSITION_DELETE_SCHEMA.accessorForField(MetadataColumns.DELETE_FILE_POS.fieldId());

  @Param({"FULL", "MEDIUM", "SHORT", "SPARSE_50", "NONE"})
  private String distribution;

  @Param({"1", "4", "64"})
  private int paths;

  private List<StructLike> records;

  @Setup(Level.Trial)
  public void setupTrial() {
    records = buildRecords(distribution, TOTAL_POSITIONS, paths);
  }

  @Benchmark
  @Threads(1)
  public void coalescedToPositionIndexes(Blackhole blackhole) {
    CharSequenceMap<PositionDeleteIndex> indexes =
        Deletes.toPositionIndexes(CloseableIterable.withNoopClose(records));
    blackhole.consume(indexes);
  }

  @Benchmark
  @Threads(1)
  public void naivePerPositionBaseline(Blackhole blackhole) {
    CharSequenceMap<PositionDeleteIndex> indexes = CharSequenceMap.create();
    try (CloseableIterable<StructLike> deletes = CloseableIterable.withNoopClose(records)) {
      for (StructLike delete : deletes) {
        CharSequence filePath = (CharSequence) FILENAME_ACCESSOR.get(delete);
        long position = (long) POSITION_ACCESSOR.get(delete);
        PositionDeleteIndex index =
            indexes.computeIfAbsent(
                filePath, key -> new BitmapPositionDeleteIndex((DeleteFile) null));
        index.delete(position);
      }
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to close position delete source", e);
    }
    blackhole.consume(indexes);
  }

  private static List<StructLike> buildRecords(String dist, int total, int numPaths) {
    int perPath = total / numPaths;
    List<StructLike> rows = Lists.newArrayListWithCapacity(perPath * numPaths);
    for (int p = 0; p < numPaths; p++) {
      String path = String.format(Locale.ROOT, "s3://bucket/warehouse/data-file-%05d.parquet", p);
      long[] positions = generate(dist, perPath);
      for (long pos : positions) {
        rows.add(new PosDeleteRow(path, pos));
      }
    }
    return rows;
  }

  private static long[] generate(String dist, int total) {
    switch (dist) {
      case "FULL":
        return PositionDistributions.contiguous(total);
      case "MEDIUM":
        return PositionDistributions.runsOf(total, 64);
      case "SHORT":
        return PositionDistributions.runsOf(total, 4);
      case "SPARSE_50":
        return PositionDistributions.mostlyContiguousWithBoundaries(total, 50);
      case "NONE":
        return PositionDistributions.strided(total, 2L);
      default:
        throw new IllegalArgumentException("Unknown distribution: " + dist);
    }
  }

  /**
   * Minimal {@link StructLike} mirroring the shape of {@code POSITION_DELETE_SCHEMA}: a {@code
   * CharSequence} file path at position 0 and a {@code long} position at position 1. Built inline
   * to avoid reaching into other modules' test sources from the JMH source set.
   */
  private static final class PosDeleteRow implements StructLike {
    private final CharSequence path;
    private final long position;

    PosDeleteRow(CharSequence path, long position) {
      this.path = path;
      this.position = position;
    }

    @Override
    public int size() {
      return 2;
    }

    @Override
    public <T> T get(int pos, Class<T> javaClass) {
      if (pos == 0) {
        return javaClass.cast(path);
      }
      if (pos == 1) {
        return javaClass.cast(position);
      }
      throw new IndexOutOfBoundsException(pos);
    }

    @Override
    public <T> void set(int pos, T value) {
      throw new UnsupportedOperationException("PosDeleteRow is immutable");
    }
  }
}
