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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TestPositionDeleteRangeConsumer {

  @Test
  void emptyInput() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(Collections.<Long>emptyList(), index);
    assertThat(index.isEmpty()).isTrue();
  }

  @Test
  void singlePosition() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(42L), index);

    assertThat(index.isDeleted(42)).isTrue();
    assertThat(index.cardinality()).isEqualTo(1);
  }

  @Test
  void consecutivePositionsCoalesced() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(10L, 11L, 12L, 13L, 14L), index);

    for (long pos = 10; pos <= 14; pos++) {
      assertThat(index.isDeleted(pos)).isTrue();
    }

    assertThat(index.isDeleted(9)).isFalse();
    assertThat(index.isDeleted(15)).isFalse();
    assertThat(index.cardinality()).isEqualTo(5);
  }

  @Test
  void multipleDisjointRanges() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(3L, 4L, 5L, 10L, 11L, 100L), index);

    assertThat(index.isDeleted(3)).isTrue();
    assertThat(index.isDeleted(4)).isTrue();
    assertThat(index.isDeleted(5)).isTrue();
    assertThat(index.isDeleted(6)).isFalse();
    assertThat(index.isDeleted(10)).isTrue();
    assertThat(index.isDeleted(11)).isTrue();
    assertThat(index.isDeleted(12)).isFalse();
    assertThat(index.isDeleted(100)).isTrue();
    assertThat(index.cardinality()).isEqualTo(6);
  }

  @Test
  void unsortedInputProducesCorrectResult() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(50L, 10L, 11L, 12L, 1L, 2L, 3L), index);

    assertThat(index.isDeleted(50)).isTrue();
    assertThat(index.isDeleted(10)).isTrue();
    assertThat(index.isDeleted(11)).isTrue();
    assertThat(index.isDeleted(12)).isTrue();
    assertThat(index.isDeleted(1)).isTrue();
    assertThat(index.isDeleted(2)).isTrue();
    assertThat(index.isDeleted(3)).isTrue();
    assertThat(index.cardinality()).isEqualTo(7);
  }

  @Test
  void duplicatePositions() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(5L, 5L, 6L, 6L), index);

    assertThat(index.isDeleted(5)).isTrue();
    assertThat(index.isDeleted(6)).isTrue();
    assertThat(index.cardinality()).isEqualTo(2);
  }

  @Test
  void largeConsecutiveRange() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    int rangeSize = 100_000;
    long[] positions = new long[rangeSize];
    for (int i = 0; i < rangeSize; i++) {
      positions[i] = i;
    }

    PositionDeleteRangeConsumer.forEach(asList(positions), index);

    assertThat(index.cardinality()).isEqualTo(rangeSize);
    assertThat(index.isDeleted(0)).isTrue();
    assertThat(index.isDeleted(rangeSize - 1)).isTrue();
    assertThat(index.isDeleted(rangeSize)).isFalse();
  }

  @Test
  void positionZero() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(0L, 1L, 2L), index);

    assertThat(index.isDeleted(0)).isTrue();
    assertThat(index.isDeleted(1)).isTrue();
    assertThat(index.isDeleted(2)).isTrue();
    assertThat(index.cardinality()).isEqualTo(3);
  }

  @Test
  void alternatingPositionsNoCoalescing() {
    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(0L, 2L, 4L, 6L), index);

    assertThat(index.isDeleted(0)).isTrue();
    assertThat(index.isDeleted(1)).isFalse();
    assertThat(index.isDeleted(2)).isTrue();
    assertThat(index.isDeleted(3)).isFalse();
    assertThat(index.isDeleted(4)).isTrue();
    assertThat(index.isDeleted(5)).isFalse();
    assertThat(index.isDeleted(6)).isTrue();
    assertThat(index.cardinality()).isEqualTo(4);
  }

  @Test
  void largeSparseInputTakesNaivePath() {
    // 4096 alternating positions -- sniff sees 100% boundaries on the 256-long prefix and
    // dispatches to the naive path. The result must still be identical to coalescing.
    int count = 4096;
    long[] positions = new long[count];
    for (int i = 0; i < count; i++) {
      positions[i] = i * 2L;
    }

    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(positions), index);

    assertThat(index.cardinality()).isEqualTo(count);
    for (int i = 0; i < count; i++) {
      assertThat(index.isDeleted(positions[i])).isTrue();
      assertThat(index.isDeleted(positions[i] + 1)).isFalse();
    }
  }

  @Test
  void sparsePrefixWithDenseTail() {
    // First 512 positions alternate, then 2048 consecutive positions. The sniff sees a sparse
    // prefix and picks the naive path; the dense tail is still emitted correctly.
    int prefixCount = 512;
    int tailCount = 2048;
    long[] positions = new long[prefixCount + tailCount];
    for (int i = 0; i < prefixCount; i++) {
      positions[i] = i * 2L;
    }

    long tailStart = 10_000L;
    for (int i = 0; i < tailCount; i++) {
      positions[prefixCount + i] = tailStart + i;
    }

    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(positions), index);

    assertThat(index.cardinality()).isEqualTo(prefixCount + tailCount);
    for (int i = 0; i < prefixCount; i++) {
      assertThat(index.isDeleted(positions[i])).isTrue();
    }

    for (int i = 0; i < tailCount; i++) {
      assertThat(index.isDeleted(tailStart + i)).isTrue();
    }

    assertThat(index.isDeleted(tailStart + tailCount)).isFalse();
  }

  @Test
  void densePrefixWithSparseTail() {
    // First 512 positions are contiguous, then 2048 alternating. The sniff sees a dense prefix
    // and picks the coalesce path; the sparse tail is still emitted correctly.
    int prefixCount = 512;
    int tailCount = 2048;
    long[] positions = new long[prefixCount + tailCount];
    for (int i = 0; i < prefixCount; i++) {
      positions[i] = i;
    }

    long tailStart = 10_000L;
    for (int i = 0; i < tailCount; i++) {
      positions[prefixCount + i] = tailStart + i * 2L;
    }

    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(positions), index);

    assertThat(index.cardinality()).isEqualTo(prefixCount + tailCount);
    for (int i = 0; i < prefixCount; i++) {
      assertThat(index.isDeleted(i)).isTrue();
    }

    for (int i = 0; i < tailCount; i++) {
      assertThat(index.isDeleted(tailStart + i * 2L)).isTrue();
      assertThat(index.isDeleted(tailStart + i * 2L + 1)).isFalse();
    }
  }

  @Test
  void lengthAtExactSniffBoundary() {
    // Input length matches the sniff window exactly: the prefix fills, the sniff runs, and the
    // streaming tail loop never executes. Both dispatch decisions must produce correct output.
    int count = 256;
    long[] dense = new long[count];
    long[] sparse = new long[count];
    for (int i = 0; i < count; i++) {
      dense[i] = i;
      sparse[i] = i * 2L;
    }

    BitmapPositionDeleteIndex denseIndex = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(dense), denseIndex);
    assertThat(denseIndex.cardinality()).isEqualTo(count);
    assertThat(denseIndex.isDeleted(0)).isTrue();
    assertThat(denseIndex.isDeleted(count - 1)).isTrue();

    BitmapPositionDeleteIndex sparseIndex = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(sparse), sparseIndex);
    assertThat(sparseIndex.cardinality()).isEqualTo(count);
    for (int i = 0; i < count; i++) {
      assertThat(sparseIndex.isDeleted(i * 2L)).isTrue();
    }
  }

  @Test
  void smallAdversarialInputSkipsSniff() {
    // Alternating input shorter than the sniff window -- the prefix never fills, so the sniff
    // is skipped and we coalesce directly. Verifies correctness on the small-list fast path.
    int count = 100;
    long[] positions = new long[count];
    for (int i = 0; i < count; i++) {
      positions[i] = i * 2L;
    }

    BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
    PositionDeleteRangeConsumer.forEach(asList(positions), index);

    assertThat(index.cardinality()).isEqualTo(count);
    for (int i = 0; i < count; i++) {
      assertThat(index.isDeleted(i * 2L)).isTrue();
      assertThat(index.isDeleted(i * 2L + 1)).isFalse();
    }
  }

  private static List<Long> asList(long... positions) {
    return Arrays.stream(positions).boxed().collect(Collectors.toList());
  }
}
