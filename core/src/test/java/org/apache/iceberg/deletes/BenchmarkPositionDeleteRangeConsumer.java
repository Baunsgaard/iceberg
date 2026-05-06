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

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Manual benchmark comparing per-position delete() vs PositionDeleteRangeConsumer.
 *
 * <p>Run with: <code>
 *   ./gradlew :iceberg-core:test --tests
 *       "org.apache.iceberg.deletes.BenchmarkPositionDeleteRangeConsumer" -Dbenchmark=true
 * </code>
 */
@EnabledIfSystemProperty(named = "benchmark", matches = "true")
public class BenchmarkPositionDeleteRangeConsumer {

  private static final int TOTAL_POSITIONS = 5_000_000;
  private static final int WARMUP_ITERATIONS = 15;
  private static final int MEASURED_ITERATIONS = 41;
  private static final int TRIM = MEASURED_ITERATIONS / 4;

  @Test
  public void runAll() {
    sanityCheck();
    System.out.println("Sanity checks passed.");

    long[][] scenarios = {
      generateNoRuns(),
      generateRuns(4, 4),
      generateRuns(64, 8),
      generateFullRun(),
      generateSparse(TOTAL_POSITIONS, 0.05, 1),
      generateSparse(TOTAL_POSITIONS, 0.50, 2),
      generateSparse(TOTAL_POSITIONS, 0.95, 3),
    };
    String[] labels = {
      "NONE (step=2)",
      "SHORT (4+gap)",
      "MEDIUM (64+gap)",
      "FULL (contiguous)",
      "SPARSE_5 (5%)",
      "SPARSE_50 (50%)",
      "SPARSE_95 (95%)",
    };

    System.out.println();
    System.out.println("=== Direct insertion (Iterable<Long> overload) ===");
    printHeader();
    for (int s = 0; s < scenarios.length; s++) {
      benchmark(labels[s], scenarios[s]);
    }
  }

  private static void printHeader() {
    System.out.printf(
        "%-16s %8s %8s %10s   %8s %8s %10s   %8s %8s   %-6s %s%n",
        "Scenario",
        "Bl.mean",
        "Co.mean",
        "mean.x",
        "Bl.min",
        "Co.min",
        "min.x",
        "Bl.std",
        "Co.std",
        "N",
        "raw(ms)");
    System.out.println("-".repeat(130));
  }

  private void benchmark(String label, long[] positions) {
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
      runBaseline(positions);
      runRangeConsumer(positions);
    }

    double[] baselineTimes = new double[MEASURED_ITERATIONS];
    double[] consumerTimes = new double[MEASURED_ITERATIONS];

    for (int i = 0; i < MEASURED_ITERATIONS; i++) {
      if (i % 2 == 0) {
        baselineTimes[i] = runBaseline(positions);
        consumerTimes[i] = runRangeConsumer(positions);
      } else {
        consumerTimes[i] = runRangeConsumer(positions);
        baselineTimes[i] = runBaseline(positions);
      }
    }

    printResult(label, positions.length, baselineTimes, consumerTimes);
  }

  private static void printResult(
      String label, int n, double[] baselineTimes, double[] consumerTimes) {
    double[] blSorted = baselineTimes.clone();
    double[] coSorted = consumerTimes.clone();
    Arrays.sort(blSorted);
    Arrays.sort(coSorted);

    double baselineMs = trimmedMean(blSorted, TRIM);
    double consumerMs = trimmedMean(coSorted, TRIM);
    double baselineStd = trimmedStdev(blSorted, TRIM);
    double consumerStd = trimmedStdev(coSorted, TRIM);
    double meanSpeedup = baselineMs / consumerMs;

    // min of the post-trim window: least noisy single observation, removing outliers caused by
    // GC or scheduling jitter. Speedup from mins tracks the steady-state ratio more tightly than
    // speedup from means across passes.
    double baselineMin = blSorted[TRIM];
    double consumerMin = coSorted[TRIM];
    double minSpeedup = baselineMin / consumerMin;

    StringBuilder rawCo = new StringBuilder();
    for (int i = 0; i < coSorted.length; i++) {
      if (i > 0) {
        rawCo.append(" ");
      }
      boolean trimmed = (i < TRIM || i >= coSorted.length - TRIM);
      if (trimmed) {
        rawCo.append(String.format(Locale.ROOT, "(%.1f)", coSorted[i]));
      } else {
        rawCo.append(String.format(Locale.ROOT, "%.1f", coSorted[i]));
      }
    }

    System.out.printf(
        Locale.ROOT,
        "%-16s %8.2f %8.2f %9.2fx   %8.2f %8.2f %9.2fx   %8.2f %8.2f   %-6s %s%n",
        label,
        baselineMs,
        consumerMs,
        meanSpeedup,
        baselineMin,
        consumerMin,
        minSpeedup,
        baselineStd,
        consumerStd,
        formatCount(n),
        rawCo);
  }

  private static String formatCount(int n) {
    if (n >= 1_000_000) {
      return (n / 1_000_000) + "M";
    } else if (n >= 1_000) {
      return (n / 1_000) + "K";
    }
    return String.valueOf(n);
  }

  private static double trimmedMean(double[] sorted, int trim) {
    double sum = 0;
    int count = sorted.length - 2 * trim;
    for (int i = trim; i < sorted.length - trim; i++) {
      sum += sorted[i];
    }
    return sum / count;
  }

  private static double trimmedStdev(double[] sorted, int trim) {
    double mean = trimmedMean(sorted, trim);
    double sumSq = 0;
    int count = sorted.length - 2 * trim;
    for (int i = trim; i < sorted.length - trim; i++) {
      double d = sorted[i] - mean;
      sumSq += d * d;
    }
    return Math.sqrt(sumSq / count);
  }

  private double runBaseline(long[] positions) {
    // Mirror the exact pre-change call site: deletes.forEach(positionDeleteIndex::delete) where
    // deletes is a CloseableIterable<Long>. This keeps the baseline apples-to-apples with the
    // consumer path -- both pay the Long unboxing cost of iterating through Iterable<Long>
    // instead of over the underlying long[].
    PositionDeleteIndex index = new BitmapPositionDeleteIndex();
    long start = System.nanoTime();
    asLongList(positions).forEach(index::delete);
    long elapsed = System.nanoTime() - start;
    consumeIndex(index);
    return elapsed / 1_000_000.0;
  }

  private double runRangeConsumer(long[] positions) {
    PositionDeleteIndex index = new BitmapPositionDeleteIndex();
    long start = System.nanoTime();
    PositionDeleteRangeConsumer.forEach(asLongList(positions), index);
    long elapsed = System.nanoTime() - start;
    consumeIndex(index);
    return elapsed / 1_000_000.0;
  }

  @SuppressWarnings("ResultOfMethodCallIgnored")
  private static void consumeIndex(PositionDeleteIndex index) {
    index.cardinality();
  }

  private static long[] generateNoRuns() {
    long[] positions = new long[TOTAL_POSITIONS];
    for (int i = 0; i < TOTAL_POSITIONS; i++) {
      positions[i] = (long) i * 2;
    }
    return positions;
  }

  private static long[] generateRuns(int runLength, int gapSize) {
    long[] positions = new long[TOTAL_POSITIONS];
    long pos = 0;
    int idx = 0;
    Random random = new Random(42);
    while (idx < TOTAL_POSITIONS) {
      int thisRun = Math.min(runLength + random.nextInt(runLength), TOTAL_POSITIONS - idx);
      for (int r = 0; r < thisRun && idx < TOTAL_POSITIONS; r++) {
        positions[idx++] = pos++;
      }
      pos += gapSize + random.nextInt(gapSize);
    }
    return positions;
  }

  private static long[] generateFullRun() {
    long[] positions = new long[TOTAL_POSITIONS];
    for (int i = 0; i < TOTAL_POSITIONS; i++) {
      positions[i] = i;
    }
    return positions;
  }

  /**
   * Exactly {@code count} distinct sorted positions uniformly sampled from a domain of {@code
   * count / density} slots, so the resulting density matches the requested value. This makes the
   * inner {@code pos - lastPosition == 1} branch track the requested density: SPARSE_50 is a true
   * ~50/50 coin flip, SPARSE_5 is adversarial (~5% hits), SPARSE_95 is best-case (~95% hits).
   *
   * <p>Uses rejection sampling for the sparse half (density <= 0.5) and the complement
   * (sampling excluded positions) for the dense half to stay fast at high densities.
   */
  private static long[] generateSparse(int count, double density, long seed) {
    long domain = (long) (count / density);
    Random random = new Random(seed);
    long[] positions;
    if (count * 2L <= domain) {
      java.util.HashSet<Long> chosen = new java.util.HashSet<>(count * 2);
      while (chosen.size() < count) {
        chosen.add((long) (random.nextDouble() * domain));
      }

      positions = new long[count];
      int idx = 0;
      for (long p : chosen) {
        positions[idx++] = p;
      }
    } else {
      long excludeCount = domain - count;
      java.util.HashSet<Long> excluded = new java.util.HashSet<>((int) (excludeCount * 2));
      while (excluded.size() < excludeCount) {
        excluded.add((long) (random.nextDouble() * domain));
      }

      positions = new long[count];
      int idx = 0;
      for (long i = 0; i < domain; i++) {
        if (!excluded.contains(i)) {
          positions[idx++] = i;
        }
      }
    }

    Arrays.sort(positions);
    return positions;
  }

  private static void sanityCheck() {
    // Iterable path (seeded)
    check("empty", new long[] {}, 0);
    check("single", new long[] {42}, 1);
    check("consecutive", new long[] {10, 11, 12}, 3);
    check("disjoint", new long[] {3, 4, 5, 10, 11, 100}, 6);
    check("unsorted", new long[] {50, 10, 11, 12, 1, 2, 3}, 7);
    check("zero-single", new long[] {0}, 1);
    check("zero-range", new long[] {0, 1, 2}, 3);
    check("zero-gap", new long[] {0, 2, 4, 6}, 4);
    check("duplicates", new long[] {5, 5, 6, 6}, 2);
  }

  private static void check(String name, long[] positions, long expectedCardinality) {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asLongList(positions), index);
    verify(name, index, positions, expectedCardinality);
  }

  private static void verify(
      String name, PositionDeleteIndex index, long[] positions, long expectedCardinality) {
    if (index.cardinality() != expectedCardinality) {
      throw new AssertionError(
          name + ": expected cardinality " + expectedCardinality + " but got " + index.cardinality());
    }

    for (long pos : positions) {
      if (!index.isDeleted(pos)) {
        throw new AssertionError(name + ": position " + pos + " should be deleted");
      }
    }
  }

  private static Iterable<Long> asLongList(long[] array) {
    return new java.util.AbstractList<Long>() {
      @Override
      public Long get(int i) {
        return array[i];
      }

      @Override
      public int size() {
        return array.length;
      }
    };
  }
}
