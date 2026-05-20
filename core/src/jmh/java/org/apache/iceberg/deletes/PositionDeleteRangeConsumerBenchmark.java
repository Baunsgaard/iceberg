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

import java.util.AbstractList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
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
 * Microbenchmark for {@link PositionDeleteRangeConsumer#forEach}. Each invocation streams a
 * pre-built {@code List<Long>} of {@value #TOTAL_POSITIONS} positions through a fresh {@link
 * BitmapPositionDeleteIndex} so the measurement covers both the coalesce/per-position dispatch in
 * the consumer and the resulting bitmap insertions.
 *
 * <p>The {@code distribution} parameter selects the run shape:
 *
 * <ul>
 *   <li>{@code FULL} -- every position is consecutive (one range covers everything).
 *   <li>{@code MEDIUM} -- ~64-position runs separated by a 1-position gap.
 *   <li>{@code SHORT} -- ~4-position runs separated by a 1-position gap.
 *   <li>{@code SPARSE_95} / {@code SPARSE_50} / {@code SPARSE_5} -- the named percentage of
 *       adjacent pairs are gaps; the remainder are consecutive.
 *   <li>{@code NONE} -- every other position (step 2), no consecutive pairs at all.
 * </ul>
 *
 * <p>Two consumer strategies are measured side by side:
 *
 * <ul>
 *   <li>The production {@link PositionDeleteRangeConsumer}, which sniffs the first {@code 256}
 *       positions and escapes to a per-position fallback if more than {@code 30%} are gaps.
 *   <li>A greedy baseline ({@link GreedyRangeConsumer}, defined in this file) that always tries to
 *       coalesce -- no sniff window, no escape. Its purpose is to isolate the cost the escape path
 *       avoids on fragmented input and to confirm that adaptive coalescing pays for itself.
 * </ul>
 *
 * <p>To run: <code>
 *   ./gradlew :iceberg-core:jmh
 *       -PjmhIncludeRegex=PositionDeleteRangeConsumerBenchmark
 *       -PjmhOutputPath=benchmark/position-delete-range-consumer-benchmark.txt
 * </code>
 */
@Fork(1)
@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Timeout(time = 5, timeUnit = TimeUnit.MINUTES)
public class PositionDeleteRangeConsumerBenchmark {

  private static final int TOTAL_POSITIONS = 5_000_000;
  // Matches PositionDeleteRangeConsumer.FOREACH_BATCH_SIZE so the boxed-iterable comparison stays
  // apples-to-apples: same drain buffer, only the consumer's coalesce strategy differs.
  private static final int FOREACH_BATCH_SIZE = 64;

  @Param({"FULL", "MEDIUM", "SHORT", "SPARSE_95", "SPARSE_50", "SPARSE_5", "NONE"})
  private String distribution;

  private List<Long> positions;
  private long[] rawPositions;

  @Setup(Level.Trial)
  public void setupTrial() {
    rawPositions = generate(distribution, TOTAL_POSITIONS);
    positions = boxedView(rawPositions);
  }

  @Benchmark
  @Threads(1)
  public void forEachIntoFreshIndex(Blackhole blackhole) {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex(ImmutableList.of());
    PositionDeleteRangeConsumer.forEach(positions, index);
    blackhole.consume(index);
  }

  /**
   * Direct call into the bulk acceptAll path. Models a caller that already has the positions in a
   * primitive {@code long[]} (e.g. a column reader after an extraction pass).
   */
  @Benchmark
  @Threads(1)
  public void acceptAllIntoFreshIndex(Blackhole blackhole) {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex(ImmutableList.of());
    PositionDeleteRangeConsumer acc = new PositionDeleteRangeConsumer(index);
    acc.acceptAll(rawPositions, 0, rawPositions.length);
    acc.flush();
    blackhole.consume(index);
  }

  /**
   * Greedy-only baseline of {@link #forEachIntoFreshIndex}: same boxed source, same batch drain,
   * but a coalescer with no sniff window and no escape. On dense distributions this should match
   * the production consumer closely; on {@code SPARSE_5}/{@code NONE} it carries the bookkeeping
   * cost that the escape path avoids.
   */
  @Benchmark
  @Threads(1)
  public void greedyForEachIntoFreshIndex(Blackhole blackhole) {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex(ImmutableList.of());
    GreedyRangeConsumer acc = new GreedyRangeConsumer(index);
    long[] buffer = new long[FOREACH_BATCH_SIZE];
    int filled = 0;
    for (Long pos : positions) {
      buffer[filled++] = pos;
      if (filled == FOREACH_BATCH_SIZE) {
        acc.acceptAll(buffer, 0, FOREACH_BATCH_SIZE);
        filled = 0;
      }
    }
    if (filled > 0) {
      acc.acceptAll(buffer, 0, filled);
    }
    acc.flush();
    blackhole.consume(index);
  }

  /** Greedy-only baseline of {@link #acceptAllIntoFreshIndex} -- direct {@code long[]} feed. */
  @Benchmark
  @Threads(1)
  public void greedyAcceptAllIntoFreshIndex(Blackhole blackhole) {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex(ImmutableList.of());
    GreedyRangeConsumer acc = new GreedyRangeConsumer(index);
    acc.acceptAll(rawPositions, 0, rawPositions.length);
    acc.flush();
    blackhole.consume(index);
  }

  private static long[] generate(String dist, int total) {
    switch (dist) {
      case "FULL":
        return contiguous(total);
      case "MEDIUM":
        return runsOf(total, 64);
      case "SHORT":
        return runsOf(total, 4);
      case "SPARSE_95":
        return mostlyContiguousWithBoundaries(total, 95);
      case "SPARSE_50":
        return mostlyContiguousWithBoundaries(total, 50);
      case "SPARSE_5":
        return mostlyContiguousWithBoundaries(total, 5);
      case "NONE":
        return strided(total, 2L);
      default:
        throw new IllegalArgumentException("Unknown distribution: " + dist);
    }
  }

  private static long[] contiguous(int total) {
    long[] out = new long[total];
    for (int i = 0; i < total; i++) {
      out[i] = i;
    }
    return out;
  }

  private static long[] strided(int total, long step) {
    long[] out = new long[total];
    for (int i = 0; i < total; i++) {
      out[i] = i * step;
    }
    return out;
  }

  // Emits runs of length runLen separated by a 1-position gap, deterministic.
  private static long[] runsOf(int total, int runLen) {
    long[] out = new long[total];
    long pos = 0;
    int written = 0;
    while (written < total) {
      int chunk = Math.min(runLen, total - written);
      for (int i = 0; i < chunk; i++) {
        out[written++] = pos++;
      }
      pos++; // gap
    }
    return out;
  }

  // Emits a sequence whose adjacent pairs cross a gap with the given probability percentage.
  // A linear-congruential-style counter walks through the values to avoid Random's allocation
  // and to keep the layout fully deterministic across runs.
  private static long[] mostlyContiguousWithBoundaries(int total, int gapPercent) {
    long[] out = new long[total];
    long pos = 0;
    out[0] = pos;
    long counter = 1;
    for (int i = 1; i < total; i++) {
      // Pseudo-random in [0, 100), seeded only by the counter -- run-to-run identical.
      counter = counter * 6364136223846793005L + 1442695040888963407L;
      int draw = (int) ((counter >>> 33) % 100);
      pos = (draw < gapPercent) ? pos + 2 : pos + 1;
      out[i] = pos;
    }
    return out;
  }

  // Wraps a long[] as a List<Long> without copying. The boxing happens in get(int) on every
  // iteration -- consistent with the live shape passed to PositionDeleteRangeConsumer.forEach
  // from Deletes.toPositionIndex (which iterates a CloseableIterable<Long>).
  private static List<Long> boxedView(long[] raw) {
    return new AbstractList<Long>() {
      @Override
      public Long get(int index) {
        return raw[index];
      }

      @Override
      public int size() {
        return raw.length;
      }
    };
  }

  /**
   * Greedy-only baseline. Maintains a single active run and either extends it or emits it on every
   * position. Mirrors {@link PositionDeleteRangeConsumer} minus the sniff window and the per-
   * position escape path; intended only as a benchmark counterfactual, not for production use.
   */
  static final class GreedyRangeConsumer {
    private final PositionDeleteIndex target;
    private boolean hasRun;
    private long rangeStart;
    private long lastPosition;

    GreedyRangeConsumer(PositionDeleteIndex target) {
      this.target = target;
    }

    void acceptAll(long[] positions, int from, int to) {
      int cursor = from;
      if (!hasRun && cursor < to) {
        long first = positions[cursor++];
        rangeStart = first;
        lastPosition = first;
        hasRun = true;
      }
      while (cursor < to) {
        long pos = positions[cursor++];
        if (pos - lastPosition != 1) {
          emit();
          rangeStart = pos;
        }
        lastPosition = pos;
      }
    }

    void flush() {
      if (hasRun) {
        emit();
        hasRun = false;
      }
    }

    private void emit() {
      if (rangeStart == lastPosition) {
        target.delete(rangeStart);
      } else {
        target.delete(rangeStart, lastPosition + 1);
      }
    }
  }
}
