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

import java.util.List;
import java.util.concurrent.TimeUnit;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
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
 * Micro-benchmark for {@code PositionDeleteRangeConsumer.forEach}-style flows over an {@link
 * Iterable Iterable&lt;Long&gt;} source. Compares three strategies for the iterator caller:
 *
 * <ul>
 *   <li>{@code noDrain} -- hand each position to {@link
 *       PositionDeleteRangeConsumer#acceptAll(long[], int, int)} as a 1-element slice, no
 *       amortization across a buffer.
 *   <li>{@code drainSmall} -- buffer to {@code long[64]} then {@code acceptAll}
 *   <li>{@code drainLarge} -- buffer to {@code long[1024]} then {@code acceptAll}
 * </ul>
 *
 * <p>The bulk-array case (Arrow) is exercised by {@code BaseDeleteLoaderBenchmark}; this one
 * isolates the boxed-source path so we can decide whether a drain ever earns its keep when boxing
 * dominates the per-element cost.
 *
 * <p>To run: <code>
 *   ./gradlew :iceberg-core:jmh
 *       -PjmhIncludeRegex=ForEachDrainBenchmark
 *       -PjmhOutputPath=benchmark/foreach-drain-benchmark.txt
 * </code>
 */
@Fork(1)
@State(Scope.Benchmark)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Timeout(time = 5, timeUnit = TimeUnit.MINUTES)
public class ForEachDrainBenchmark {

  private static final int POSITIONS = 1_000_000;
  private static final int SMALL_BUFFER = 64;
  private static final int LARGE_BUFFER = 1024;

  @Param({"dense", "sparse"})
  private String distribution;

  private List<Long> boxedPositions;

  @Setup
  public void setupBenchmark() {
    long[] raw =
        "dense".equals(distribution)
            ? PositionDistributions.contiguous(POSITIONS)
            : PositionDistributions.randomSorted(POSITIONS, 42L);
    List<Long> list = Lists.newArrayListWithCapacity(POSITIONS);
    for (long pos : raw) {
      list.add(pos);
    }
    boxedPositions = list;
  }

  // Models a caller that hands every position to acceptAll as a 1-element slice. Equivalent to
  // the per-element path we used to expose via accept(long): same state-machine dispatch on every
  // position, no amortization across a buffer.
  @Benchmark
  @Threads(1)
  public void noDrain(Blackhole bh) {
    BitmapPositionDeleteIndex target = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer acc = new PositionDeleteRangeConsumer(target);
    long[] single = new long[1];
    for (Long pos : boxedPositions) {
      single[0] = pos;
      acc.acceptAll(single, 0, 1);
    }
    acc.flush();
    bh.consume(target);
  }

  @Benchmark
  @Threads(1)
  public void drainSmall(Blackhole bh) {
    BitmapPositionDeleteIndex target = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer acc = new PositionDeleteRangeConsumer(target);
    long[] buffer = new long[SMALL_BUFFER];
    int filled = 0;
    for (Long pos : boxedPositions) {
      buffer[filled++] = pos;
      if (filled == SMALL_BUFFER) {
        acc.acceptAll(buffer, 0, SMALL_BUFFER);
        filled = 0;
      }
    }
    if (filled > 0) {
      acc.acceptAll(buffer, 0, filled);
    }
    acc.flush();
    bh.consume(target);
  }

  @Benchmark
  @Threads(1)
  public void drainLarge(Blackhole bh) {
    BitmapPositionDeleteIndex target = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer acc = new PositionDeleteRangeConsumer(target);
    long[] buffer = new long[LARGE_BUFFER];
    int filled = 0;
    for (Long pos : boxedPositions) {
      buffer[filled++] = pos;
      if (filled == LARGE_BUFFER) {
        acc.acceptAll(buffer, 0, LARGE_BUFFER);
        filled = 0;
      }
    }
    if (filled > 0) {
      acc.acceptAll(buffer, 0, filled);
    }
    acc.flush();
    bh.consume(target);
  }
}
