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
import java.util.concurrent.TimeUnit;
import org.apache.iceberg.Files;
import org.apache.iceberg.MetadataColumns;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Schema;
import org.apache.iceberg.StructLike;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetReaders;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.deletes.Deletes;
import org.apache.iceberg.deletes.PositionDelete;
import org.apache.iceberg.deletes.PositionDeleteIndex;
import org.apache.iceberg.deletes.PositionDeleteWriter;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.parquet.Parquet;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
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
 * A benchmark that compares the record-based path for materializing a {@link PositionDeleteIndex}
 * against the zero-copy Arrow path provided by {@link VectorizedPositionDeleteReader}.
 *
 * <p>Two distributions are exercised: {@code dense} (every position in {@code [0, N)}, which
 * coalesces into a single range) and {@code sparse} (every {@code STRIDE}-th position, which never
 * coalesces). Both filtered (single data file) and no-filter (DV-style) reads are measured.
 *
 * <p>To run this benchmark: <code>
 *   ./gradlew :iceberg-arrow:jmh
 *       -PjmhIncludeRegex=PositionDeleteReaderBenchmark
 *       -PjmhOutputPath=benchmark/position-delete-reader-benchmark.txt
 * </code>
 */
@Fork(1)
@State(Scope.Benchmark)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@BenchmarkMode(Mode.SingleShotTime)
@Timeout(time = 5, timeUnit = TimeUnit.MINUTES)
public class PositionDeleteReaderBenchmark {

  private static final String DATA_LOCATION = "s3://bucket/path/to/data-file.parquet";
  private static final int NUM_POSITIONS = 2_000_000;
  private static final long SPARSE_STRIDE = 100L;

  private static final Schema FULL_SCHEMA =
      new Schema(MetadataColumns.DELETE_FILE_PATH, MetadataColumns.DELETE_FILE_POS);
  private static final Schema POS_ONLY_SCHEMA = new Schema(MetadataColumns.DELETE_FILE_POS);

  @Param({"dense", "sparse"})
  private String distribution;

  private File deleteFile;

  @Setup
  public void setup() throws IOException {
    deleteFile = java.nio.file.Files.createTempFile("position-delete-bench", ".parquet").toFile();
    deleteFile.delete();

    long[] positions = "dense".equals(distribution) ? densePositions() : sparsePositions();
    writeDeleteFile(deleteFile, positions);
  }

  @TearDown
  public void tearDown() {
    if (deleteFile != null) {
      deleteFile.delete();
    }
  }

  @Benchmark
  @Threads(1)
  public void readBaselineFiltered(Blackhole blackhole) throws IOException {
    blackhole.consume(readWithRecordPath(deleteFile, DATA_LOCATION));
  }

  @Benchmark
  @Threads(1)
  public void readVectorizedFiltered(Blackhole blackhole) {
    blackhole.consume(
        VectorizedPositionDeleteReader.read(Files.localInput(deleteFile), DATA_LOCATION, null));
  }

  @Benchmark
  @Threads(1)
  public void readBaselineNoFilter(Blackhole blackhole) throws IOException {
    blackhole.consume(readWithRecordPath(deleteFile, null));
  }

  @Benchmark
  @Threads(1)
  public void readVectorizedNoFilter(Blackhole blackhole) {
    blackhole.consume(
        VectorizedPositionDeleteReader.read(Files.localInput(deleteFile), null, null));
  }

  /**
   * Mirrors the engine-side path: open the Parquet delete file as records, then build a {@link
   * PositionDeleteIndex}. The filtered branch reuses {@link Deletes#deletePositions(CharSequence,
   * CloseableIterable)} + {@link Deletes#toPositionIndex(CloseableIterable)}; the no-filter branch
   * projects pos only and inserts each position individually, which is the baseline a DV-style
   * reader would have today.
   */
  private static PositionDeleteIndex readWithRecordPath(File file, CharSequence dataLocation)
      throws IOException {
    Schema projection = dataLocation == null ? POS_ONLY_SCHEMA : FULL_SCHEMA;
    CloseableIterable<Record> records =
        Parquet.read(Files.localInput(file))
            .project(projection)
            .createReaderFunc(
                fileSchema -> GenericParquetReaders.buildReader(projection, fileSchema))
            .build();

    if (dataLocation != null) {
      @SuppressWarnings({"unchecked", "rawtypes"})
      CloseableIterable<StructLike> rows = (CloseableIterable) records;
      CloseableIterable<Long> positions = Deletes.deletePositions(dataLocation, rows);
      return Deletes.toPositionIndex(positions);
    }

    PositionDeleteIndex index = PositionDeleteIndex.create();
    try (CloseableIterable<Record> iterable = records) {
      for (Record record : iterable) {
        index.delete((long) record.get(0));
      }
    }
    return index;
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
}
