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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Performance benchmark comparing setRange() (native bulk range add) vs a loop of set() calls (the
 * pre-optimization approach).
 *
 * <p>Skipped in CI. Run manually with:
 *
 * <pre>{@code
 * ./gradlew :iceberg-core:test \
 *   --tests 'org.apache.iceberg.deletes.TestRoaringPositionBitmapPerf' \
 *   -Dbenchmark=true
 * }</pre>
 */
public class TestRoaringPositionBitmapPerf {

  private static final int WARMUP_ITERATIONS = 3;
  private static final int MEASURE_ITERATIONS = 5;

  @Test
  @EnabledIfSystemProperty(named = "benchmark", matches = "true")
  public void benchmarkSetRangeVsSetLoop() {
    long[][] scenarios = {
      {0, 100},
      {0, 10_000},
      {500, 200_500},
      {0, 1_000_000},
      {(1L << 32) - 5_000, (1L << 32) + 5_000},
    };

    String[] names = {
      "small_range_100",
      "medium_range_10k",
      "large_range_200k",
      "large_range_1M",
      "cross_key_boundary_10k",
    };

    System.out.println();
    System.out.println("============================================================");
    System.out.println("  Java RoaringPositionBitmap.setRange Benchmark");
    System.out.println("  Before = loop of set() | After = native setRange()");
    System.out.println(
        "  Iterations: warmup=" + WARMUP_ITERATIONS + ", measure=" + MEASURE_ITERATIONS);
    System.out.println("============================================================");

    for (int i = 0; i < scenarios.length; i++) {
      long start = scenarios[i][0];
      long end = scenarios[i][1];
      long count = end - start;

      measureSetLoop(start, end, WARMUP_ITERATIONS);
      measureSetRange(start, end, WARMUP_ITERATIONS);

      double beforeNs = measureSetLoop(start, end, MEASURE_ITERATIONS);
      double afterNs = measureSetRange(start, end, MEASURE_ITERATIONS);
      double speedup = afterNs > 0 ? beforeNs / afterNs : Double.POSITIVE_INFINITY;

      System.out.println();
      System.out.println("--- " + names[i] + " (" + count + " positions) ---");
      System.out.println("  Before (set loop):   " + formatTime(beforeNs));
      System.out.println("  After  (setRange):   " + formatTime(afterNs));
      System.out.printf("  Speedup:             %.1fx%n", speedup);
    }

    System.out.println();
    System.out.println("============================================================");
  }

  private double measureSetLoop(long start, long end, int iterations) {
    long totalNs = 0;
    for (int i = 0; i < iterations; i++) {
      RoaringPositionBitmap bitmap = new RoaringPositionBitmap();
      long t0 = System.nanoTime();
      for (long pos = start; pos < end; pos++) {
        bitmap.set(pos);
      }
      totalNs += System.nanoTime() - t0;
      assertThat(bitmap.cardinality()).isGreaterThan(0);
    }
    return (double) totalNs / iterations;
  }

  private double measureSetRange(long start, long end, int iterations) {
    long totalNs = 0;
    for (int i = 0; i < iterations; i++) {
      RoaringPositionBitmap bitmap = new RoaringPositionBitmap();
      long t0 = System.nanoTime();
      bitmap.setRange(start, end);
      totalNs += System.nanoTime() - t0;
      assertThat(bitmap.cardinality()).isGreaterThan(0);
    }
    return (double) totalNs / iterations;
  }

  private static String formatTime(double ns) {
    if (ns < 1_000) {
      return String.format("%.1f ns", ns);
    } else if (ns < 1_000_000) {
      return String.format("%.1f us", ns / 1_000);
    } else {
      return String.format("%.2f ms", ns / 1_000_000);
    }
  }
}
