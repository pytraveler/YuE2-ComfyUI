"""The band plan of Mel-Band RoFormer, checked against the released weights without torch."""

from __future__ import annotations

import json
import pathlib

from yue2_comfy.vocals import bands

KEYS = json.loads((pathlib.Path(__file__).resolve().parent / "data" / "melbandroformer_keys.json")
                  .read_text(encoding="ascii"))


def widths_in_the_weights():
    return [KEYS["tensors"]["band_split.to_features.{}.0.gamma".format(band)][0] for band in range(bands.BANDS)]


def test_every_band_reads_exactly_as_many_numbers_as_its_weights_expect():
    """The widths are the only trace the band plan leaves in the weights, so they are the ground truth."""
    assert bands.band_widths() == widths_in_the_weights()
    assert sum(bands.band_widths()) == 7916


def test_the_mask_estimators_hand_back_two_numbers_per_number_read():
    for band, width in enumerate(bands.band_widths()):
        assert KEYS["tensors"]["mask_estimators.0.to_freqs.{}.0.4.weight".format(band)] == [2 * width, 1536]


def test_every_bin_is_read_by_some_band_and_the_average_divides_by_that_count():
    counts = bands.bands_per_bin()
    assert len(counts) == bands.N_FFT // 2 + 1
    assert min(counts) >= 1
    assert sum(counts) == sum(len(inside) for inside in bands.memberships())


def test_the_two_forced_bins_are_in_their_bands():
    """0 Hz sits on the first triangle's foot and Nyquist on the last one's; training forced both in."""
    found = bands.memberships()
    assert found[0][0] == 0
    assert found[-1][-1] == bands.N_FFT // 2


def test_the_gather_order_takes_both_channels_of_each_bin_band_by_band():
    found = bands.memberships()
    order = bands.gather_order()
    assert len(order) == bands.CHANNELS * sum(len(inside) for inside in found)
    first, second = found[0][0], found[0][1]
    assert order[:4] == [first * 2, first * 2 + 1, second * 2, second * 2 + 1]


def test_the_mel_scale_comes_back_to_the_frequency_it_left():
    for hz in (0.0, 440.0, 999.0, 1000.0, 8000.0, 22050.0):
        assert abs(bands.mel_to_hz(bands.hz_to_mel(hz)) - hz) <= 1e-6 * max(1.0, hz)
