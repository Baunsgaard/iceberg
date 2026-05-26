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
import java.util.Random;

/**
 * Position-array generators shared across the position-delete JMH benchmarks. Centralising them
 * keeps {@code DeletesToPositionIndexesBenchmark}, {@code PositionDeleteRangeConsumerBenchmark},
 * and {@code ForEachDrainBenchmark} on the same fixture shapes so benchmark numbers can be compared
 * like-for-like.
 *
 * <p>All generators are deterministic for a given input so benchmark outputs do not drift between
 * runs. The {@link #randomSorted(int, long)} helper is the lone exception by design: it seeds a
 * {@link Random} explicitly so a fixed seed reproduces the same layout.
 */
final class PositionDistributions {

  private PositionDistributions() {}

  /** {@code [0, total)} -- one contiguous run, ideal for coalescing. */
  static long[] contiguous(int total) {
    long[] out = new long[total];
    for (int i = 0; i < total; i++) {
      out[i] = i;
    }
    return out;
  }

  /**
   * Evenly-spaced positions {@code i * step} for {@code i in [0, total)}. With {@code step >= 2}
   * the result has zero consecutive pairs, exercising the per-position fallback exclusively.
   */
  static long[] strided(int total, long step) {
    long[] out = new long[total];
    for (int i = 0; i < total; i++) {
      out[i] = i * step;
    }
    return out;
  }

  /**
   * Runs of length {@code runLen} separated by a one-position gap. Yields {@code total} positions
   * regardless of how many full runs that takes. Models partially-coalesce-able input where the
   * sniff window should still favour the bulk path.
   */
  static long[] runsOf(int total, int runLen) {
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

  /**
   * Pseudo-random ascending sequence whose adjacent pairs cross a gap with probability {@code
   * gapPercent}%. A linear-congruential counter walks the sequence so the layout is identical
   * across runs without allocating a {@link Random} instance per call.
   */
  static long[] mostlyContiguousWithBoundaries(int total, int gapPercent) {
    long[] out = new long[total];
    long pos = 0;
    out[0] = pos;
    long counter = 1;
    for (int i = 1; i < total; i++) {
      counter = counter * 6364136223846793005L + 1442695040888963407L;
      int draw = (int) ((counter >>> 33) % 100);
      pos = (draw < gapPercent) ? pos + 2 : pos + 1;
      out[i] = pos;
    }
    return out;
  }

  /**
   * Random positions in a wide value space, sorted ascending. Models sparse deletes against a very
   * large file where most adjacent pairs cross a gap and a few might happen to be consecutive. The
   * seed is fixed by the caller so the sequence is reproducible.
   */
  static long[] randomSorted(int total, long seed) {
    Random random = new Random(seed);
    long[] out = new long[total];
    for (int i = 0; i < total; i++) {
      out[i] = ((long) random.nextInt(Integer.MAX_VALUE)) * 100L;
    }
    Arrays.sort(out);
    return out;
  }
}
