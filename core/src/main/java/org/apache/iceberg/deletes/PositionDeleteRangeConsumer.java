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

import java.util.Iterator;

/**
 * Coalesces consecutive position deletes into range deletes.
 *
 * <p>Consecutive positions (e.g. 3, 4, 5, 6) are accumulated and flushed as a single {@link
 * PositionDeleteIndex#delete(long, long)} range call instead of individual point deletes.
 *
 * <p>The first {@code SNIFF_SIZE} positions are processed on the coalesce path while counting
 * boundaries -- adjacent pairs where {@code pos[i] - pos[i-1] != 1}. If the observed boundary
 * density exceeds {@code BOUNDARY_THRESHOLD_PERCENT}, the remaining stream switches to a plain
 * per-position loop equivalent to the original {@code delete(pos)} behavior, avoiding the extra
 * compare-and-emit overhead on inputs where coalescing cannot amortize. Inputs shorter than the
 * sniff window skip the check and coalesce directly. The sniff is a prefix heuristic:
 * misclassifying an input with a dense head and a sparse tail (or vice versa) only costs
 * throughput, not correctness -- both paths produce identical index contents.
 */
final class PositionDeleteRangeConsumer {

  // Number of prefix positions inspected to estimate boundary density.
  private static final int SNIFF_SIZE = 256;

  // Boundary density threshold, expressed as a percentage of sniff comparisons. Inputs above
  // this threshold switch to per-position delete for the tail. Calibrated so FULL / MEDIUM /
  // SHORT / SPARSE_95 (<=20% boundaries) stay on the coalesce path and SPARSE_50 / SPARSE_5 /
  // NONE (>=50% boundaries) switch to per-position delete.
  private static final int BOUNDARY_THRESHOLD_PERCENT = 30;

  private PositionDeleteRangeConsumer() {}

  static void forEach(Iterable<Long> positions, PositionDeleteIndex target) {
    Iterator<Long> it = positions.iterator();
    if (it.hasNext() && !coalesceOrEscape(target, it)) {
      naiveTail(target, it);
    }
  }

  // Runs the coalesce loop with a prefix-boundary check. The first SNIFF_SIZE positions are
  // processed on the coalesce path while counting boundaries; if the observed density exceeds
  // the threshold, flushes the active single-element range and returns false so the caller can
  // drain the remaining stream per-position. Otherwise processes the entire input, flushes the
  // trailing range, and returns true. Caller must ensure the iterator has at least one element.
  private static boolean coalesceOrEscape(PositionDeleteIndex target, Iterator<Long> it) {
    long rangeStart = it.next();
    long lastPosition = rangeStart;
    int processed = 1;
    int boundaries = 0;

    while (processed < SNIFF_SIZE && it.hasNext()) {
      long pos = it.next();
      if (pos - lastPosition != 1) {
        boundaries++;
        emit(target, rangeStart, lastPosition);
        rangeStart = pos;
      }

      lastPosition = pos;
      processed++;
    }

    if (processed == SNIFF_SIZE
        && boundaries * 100 > (SNIFF_SIZE - 1) * BOUNDARY_THRESHOLD_PERCENT) {
      // adversarial prefix -- flush any pending range and let the caller drain the rest
      emit(target, rangeStart, lastPosition);
      return false;
    }

    while (it.hasNext()) {
      long pos = it.next();
      if (pos - lastPosition != 1) {
        emit(target, rangeStart, lastPosition);
        rangeStart = pos;
      }

      lastPosition = pos;
    }

    emit(target, rangeStart, lastPosition);
    return true;
  }

  // Tight per-position loop for the remaining iterator. Split out so the coalesce frame stays
  // small enough for the JIT to inline aggressively into the caller.
  private static void naiveTail(PositionDeleteIndex target, Iterator<Long> tail) {
    while (tail.hasNext()) {
      target.delete(tail.next());
    }
  }

  // dispatches to the cheaper single-position delete when the range is one element
  private static void emit(PositionDeleteIndex target, long rangeStart, long lastPosition) {
    if (rangeStart == lastPosition) {
      target.delete(rangeStart);
    } else {
      target.delete(rangeStart, lastPosition + 1);
    }
  }
}
