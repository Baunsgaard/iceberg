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
 * Companion to {@link DeletesToPositionIndexesBenchmark}, focused on the {@code many-paths /
 * few-positions-per-path} regime. The production path allocates a fresh {@link
 * PositionDeleteRangeConsumer}, a 64-element drain buffer, and a {@link
 * java.util.PrimitiveIterator.OfLong} wrapper per path, so when each path only contributes a handful
 * of positions, this setup cost can dominate the per-record work.
 *
 * <p>Total record count is held constant ({@value #TOTAL_RECORDS}) so each invocation processes the
 * same number of rows; we vary {@code positionsPerPath} to slide between "coalescing wins" and
 * "per-path allocation dominates". Positions inside each path are contiguous so the coalescing fast
 * path is always available -- the question being measured is whether the constant per-path setup
 * cost outweighs the per-record savings as that per-path budget shrinks.
 *
 * <p>To run: <code>
 *   ./gradlew :iceberg-core:jmh
 *       -PjmhIncludeRegex=DeletesToPositionIndexesFewPerPathBenchmark
 *       -PjmhOutputPath=benchmark/deletes-to-position-indexes-few-per-path.txt
 *       -PjmhJsonOutputPath=benchmark/deletes-to-position-indexes-few-per-path.json
 * </code>
 */
@Fork(1)
@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Timeout(time = 5, timeUnit = TimeUnit.MINUTES)
public class DeletesToPositionIndexesFewPerPathBenchmark {

  private static final int TOTAL_RECORDS = 250_000;

  // Replicates the package-private accessors in Deletes.java so the naive baseline reads the
  // record fields through the same Accessor<StructLike> dispatch the production path used to.
  private static final Schema POSITION_DELETE_SCHEMA =
      new Schema(MetadataColumns.DELETE_FILE_PATH, MetadataColumns.DELETE_FILE_POS);
  private static final Accessor<StructLike> FILENAME_ACCESSOR =
      POSITION_DELETE_SCHEMA.accessorForField(MetadataColumns.DELETE_FILE_PATH.fieldId());
  private static final Accessor<StructLike> POSITION_ACCESSOR =
      POSITION_DELETE_SCHEMA.accessorForField(MetadataColumns.DELETE_FILE_POS.fieldId());

  @Param({"1", "2", "4", "8", "16", "64", "256"})
  private int positionsPerPath;

  private List<StructLike> records;

  @Setup(Level.Trial)
  public void setupTrial() {
    records = buildRecords(TOTAL_RECORDS, positionsPerPath);
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

  /**
   * Builds {@code total} records partitioned into adjacent runs of length {@code perPath}, each run
   * carrying a distinct synthetic data-file path. Positions inside each run are {@code [0,
   * perPath)} (contiguous) -- the best case for coalescing. The number of distinct paths is {@code
   * ceil(total / perPath)}.
   */
  private static List<StructLike> buildRecords(int total, int perPath) {
    List<StructLike> rows = Lists.newArrayListWithCapacity(total);
    int pathIndex = 0;
    int written = 0;
    while (written < total) {
      String path =
          String.format(Locale.ROOT, "s3://bucket/warehouse/data-file-%07d.parquet", pathIndex++);
      int chunk = Math.min(perPath, total - written);
      for (int i = 0; i < chunk; i++) {
        rows.add(new PosDeleteRow(path, i));
        written++;
      }
    }
    return rows;
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
