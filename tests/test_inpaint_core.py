"""The edit core on a model a few megabytes big, on the CPU.

The released model is too big for CI, but nothing an edit promises depends on
its size: a YuE2 of two layers and thirty-two channels runs the same code. What
is held here is what the stand measured on the card and what the port has to
keep:

* the acoustic stage with nothing held is the song's own, to the bit, and a
  held frame comes out as the latent it was held on, to the bit;
* the noise of every frame is the row the song's own draw gave it;
* the join scores are the model's log-probabilities of the old frames, and
  feeding the context in pieces does not change what they say;
* the sound outside the decoded window is the old sound, and inside it is what
  decoding the whole edited song would have given;
* a retake and a cut change their stretch and nothing else.

The decoder here is a stand-in that hears three frames either way, the way the
real one hears its halo, so a window decoded out of the song is exactly the
same stretch of the whole song decoded.
"""

from __future__ import annotations

import math
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
core = pytest.importorskip("yue2_comfy.inpaint.core")

from yue2_comfy import generate, songs  # noqa: E402
from yue2_comfy.constants import DEFAULT_OPTIONS  # noqa: E402
from yue2_comfy.inpaint import ops  # noqa: E402
from yue2_comfy.vendor.yue2 import cuda_graph, nar, sampling  # noqa: E402
from yue2_comfy.vendor.yue2.protocol import (  # noqa: E402
    CODEC_OFFSET, CODEC_SIZE, MUSIC_END, SongRequest, token_prefixes,
)

HOP = 1920
REACH = 3
HALO = 4
FRAMES = 120
SEED = 3
SETTINGS = dict(DEFAULT_OPTIONS, ode_steps=4, offload="off", cot="full")


class Words:
    """A tokenizer that turns every character into one ordinary text token."""

    def encode(self, text):
        return [100 + ord(character) % 500 for character in text]


def decode_locally(latents):
    """A decoder that hears REACH frames either way and writes 1920 samples a frame, 64 short at the end."""
    z = latents.float()
    padded = torch.nn.functional.pad(z[:, :2].T, (REACH, REACH))
    weights = torch.arange(1, 2 * REACH + 2, dtype=torch.float32)
    values = torch.stack([(padded[:, f:f + 2 * REACH + 1] * weights).sum(-1)
                          for f in range(z.shape[0])], dim=-1)
    shape = torch.linspace(0.5, 1.0, HOP)
    audio = (values[:, :, None] * shape).reshape(2, z.shape[0] * HOP)[:, :z.shape[0] * HOP - 64]
    return (audio * 0.01).unsqueeze(0)


@pytest.fixture(autouse=True)
def local_decoder(monkeypatch):
    monkeypatch.setattr(generate, "decode", lambda models, latents, *args, **kwargs: (decode_locally(latents), {}))


@pytest.fixture(scope="module")
def lm():
    from yue2_comfy.vendor.yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

    torch.manual_seed(5)
    config = YuE2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, intermediate_size=64)
    return YuE2ForCausalLM(config).eval()


@pytest.fixture
def models(lm):
    vae = types.SimpleNamespace(config=types.SimpleNamespace(downsampling_ratio=HOP, decode_halo_frames=HALO))
    return types.SimpleNamespace(lm=lm, vae=vae, tokenizer=Words(), device="cpu", offload="off", low_vram=False)


def prompt(lyrics="[verse]\nla la la", score=(5, 6, 7)):
    request = SongRequest(style="a style", lyrics=lyrics, cot="full", seed=SEED)
    return token_prefixes(request, Words(), list(score))


def codec_of(frames=FRAMES):
    return [(index * 7919 + 13) % CODEC_SIZE for index in range(frames)]


@pytest.fixture(scope="module")
def song(lm):
    """A song of the tiny model: its latents are its own acoustic stage's, its sound the stand-in decoder's."""
    prefix, codec = prompt(), codec_of()
    latents = nar.synthesize(lm, prefix, codec, SEED, steps=4)
    sound = decode_locally(latents)
    made = songs.made("YuE2 Generate Song", "a style", "[verse]\nla la la", SEED, SETTINGS, "X:1", prefix,
                      None, codec, [[SEED, 0, FRAMES]], latents, 48000, 2, sound.shape[-1])
    return made, sound


def test_the_noise_of_a_fresh_song_is_the_draw_the_acoustic_stage_makes():
    drawn = torch.cat([chunk.noise for chunk in nar.song_chunks(prompt(), codec_of(50), SEED)])
    assert torch.equal(core.noise_of([[SEED, 0, 50]]), drawn)


def test_the_rows_of_a_long_draw_are_the_rows_of_a_short_one():
    def draw(seed, rows):
        return torch.randn((rows, 64), generator=torch.Generator().manual_seed(seed))

    assert torch.equal(draw(SEED, 13)[:5], draw(SEED, 5))
    noise = core.noise_of([[SEED, 0, 5], [9, 0, 3], [SEED, 9, 4]])
    assert torch.equal(noise, torch.cat([draw(SEED, 5), draw(9, 3), draw(SEED, 13)[9:13]]))


def test_latents_come_back_from_their_bytes(song):
    made, _sound = song
    again = core.latents_of(made)
    assert again.shape == (FRAMES, 64) and again.dtype == torch.float32
    assert songs._floats(again) == made.latents


class Engine:
    """The part of CachedNAR the solver talks to, with a velocity anyone can compute."""

    device = torch.device("cpu")
    dtype = torch.bfloat16

    def __init__(self, noise):
        self.chunk = types.SimpleNamespace(noise=noise)

    def velocity(self, state, raw_t):
        return (state * 0.3 + math.tanh(raw_t) * 0.1).to(self.dtype)


def test_a_solve_with_nothing_held_is_upstreams_to_the_bit():
    noise = torch.randn((20, 64), generator=torch.Generator().manual_seed(1))
    known = torch.zeros((20, 64))
    held = torch.zeros(20, dtype=torch.bool)
    engine = Engine(noise)
    assert torch.equal(core._solved(engine, 8, noise, known, held), nar.CachedNAR.solve(engine, 8))


def test_a_held_frame_ends_as_the_latent_it_was_held_on():
    noise = torch.randn((20, 64), generator=torch.Generator().manual_seed(1))
    known = torch.randn((20, 64), generator=torch.Generator().manual_seed(2)).bfloat16().float()
    held = torch.ones(20, dtype=torch.bool)
    held[8:12] = False
    result = core._solved(Engine(noise), 8, noise, known, held)
    assert torch.equal(result[held], known[held])
    assert not torch.equal(result[8:12], known[8:12])


def test_repaint_with_nothing_held_is_the_songs_own_acoustic_stage(models):
    prefix, codec = prompt(), codec_of()
    held = torch.zeros(FRAMES, dtype=torch.bool)
    result = core.repaint(models, prefix, codec, core.noise_of([[SEED, 0, FRAMES]]),
                          torch.zeros((FRAMES, 64)), held, 4)
    assert torch.equal(result, nar.synthesize(models.lm, prefix, codec, SEED, steps=4))


def test_repaint_keeps_held_frames_and_draws_the_rest_again(models, song):
    made, _sound = song
    known = core.latents_of(made)
    held = torch.ones(FRAMES, dtype=torch.bool)
    held[40:60] = False
    result = core.repaint(models, list(made.prefix), list(made.codec), core.noise_of(made.noise),
                          known, held, 4)
    assert torch.equal(result[held], known[held])
    assert not torch.equal(result[40:60], known[40:60])


def held_but(*runs):
    held = torch.ones(FRAMES, dtype=torch.bool)
    for low, high in runs:
        held[low:high] = False
    return held


def test_windows_reach_past_the_edit_and_share_when_close(monkeypatch):
    monkeypatch.setattr(core, "WINDOW", 10)
    assert core.windows_of(held_but((40, 60)), 1000) == [(30, 70)]
    assert core.windows_of(held_but((5, 8), (110, 118)), 1000) == [(0, 18), (100, 120)]
    assert core.windows_of(held_but((40, 45), (60, 65)), 1000) == [(30, 75)]
    assert core.windows_of(held_but((40, 45), (66, 70)), 1000) == [(30, 55), (56, 80)]
    assert core.windows_of(held_but(), 1000) == []


def test_a_window_fits_the_acoustic_stage_and_never_holds_another_windows_frames(monkeypatch):
    monkeypatch.setattr(core, "WINDOW", 10)
    assert core.windows_of(held_but((40, 50)), 16) == [(37, 53)]
    assert core.windows_of(held_but((40, 45), (50, 55)), 12) == [(37, 49), (47, 59)]
    assert core.windows_of(held_but((2, 12)), 14) == [(0, 14)]
    with pytest.raises(ValueError, match="holds at most"):
        core.windows_of(held_but((40, 60)), 12)


def test_an_edit_is_drawn_in_a_window_as_upstream_draws_a_chunk_of_it(models, song, monkeypatch):
    made, _sound = song
    prefix, codec = list(made.prefix), list(made.codec)
    monkeypatch.setattr(core, "WINDOW", 10)
    known = core.latents_of(made)
    noise = core.noise_of(made.noise)
    drawn = []
    original = nar.CachedNAR

    class Counted(original):
        def __init__(self, model, chunk, *args, **kwargs):
            drawn.append((list(chunk.ar_tokens), chunk.noise.clone()))
            super().__init__(model, chunk, *args, **kwargs)

    monkeypatch.setattr(nar, "CachedNAR", Counted)
    held = held_but((70, 80))
    result = core.repaint(models, prefix, codec, noise, known, held, 4)
    assert [len(tokens) for tokens, _noise in drawn] == [len(prefix) + 30 + 1]
    tokens, window_noise = drawn[0]
    assert tokens == prefix + [value + CODEC_OFFSET for value in codec[60:90]] + [MUSIC_END]
    assert torch.equal(window_noise, noise[60:90])
    assert torch.equal(result[held], known[held])
    assert not torch.equal(result[70:80], known[70:80])
    engine = original(models.lm, nar.Chunk(tokens, noise[60:90].contiguous()), "sdpa", None)
    assert torch.equal(result[60:90], core._solved(engine, 4, noise[60:90], known[60:90], held[60:90]))


def test_an_edit_across_the_old_chunk_boundary_is_drawn(models, song, monkeypatch):
    made, _sound = song
    prefix = list(made.prefix)
    monkeypatch.setattr(core, "CONTEXT", len(prefix) + 3 + FRAMES)
    monkeypatch.setattr(core, "WINDOW", 10)
    known = core.latents_of(made)
    across = held_but((55, 65))
    result = core.repaint(models, prefix, list(made.codec), core.noise_of(made.noise), known, across, 4)
    assert torch.equal(result[across], known[across])
    assert not torch.equal(result[55:65], known[55:65])


def test_the_sampler_is_told_the_tokens_sung_before_the_edit(monkeypatch):
    seen = []

    def spy(logits, config, history, step, phase, legacy_off=False):
        seen.append(list(history))
        return logits

    monkeypatch.setattr(sampling, "distribution", spy)
    with core._remembering([1, 2, 3]):
        sampling.distribution(None, None, [9], 0, "semantic")
    sampling.distribution(None, None, [9], 0, "semantic")
    assert seen == [[1, 2, 3, 9], [9]]


def context_of(made, start):
    return list(made.prefix) + [value + CODEC_OFFSET for value in made.codec[:start]]


def test_the_edit_sings_codec_tokens_and_the_same_ones_again_for_the_same_seed(models, song, monkeypatch):
    made, _sound = song
    context = context_of(made, 20)
    history = context[-5:]
    seen = []
    original = sampling.distribution

    def spy(logits, config, found, step, phase, legacy_off=False):
        seen.append(list(found))
        return original(logits, config, found, step, phase, legacy_off)

    monkeypatch.setattr(sampling, "distribution", spy)
    tokens, _timing, ended = core.perform(models, context, None, 16, 16, 11, history, SETTINGS)
    assert len(tokens) == 16 and not ended
    assert all(CODEC_OFFSET <= token < CODEC_OFFSET + CODEC_SIZE for token in tokens)
    assert seen[0] == history and seen[1] == history + tokens[:1]
    again, _timing, _ended = core.perform(models, context, None, 16, 16, 11, history, SETTINGS)
    other, _timing, _ended = core.perform(models, context, None, 16, 16, 12, history, SETTINGS)
    assert again == tokens and other != tokens


def test_the_model_may_end_the_song_once_it_has_sung_the_shortest_length(models, song, monkeypatch):
    made, _sound = song
    original = sampling.distribution

    def eager_to_end(logits, config, found, step, phase, legacy_off=False):
        scores = original(logits, config, found, step, phase, legacy_off)
        if step >= config.min_tokens:
            scores[..., MUSIC_END] = 1e4
        return scores

    monkeypatch.setattr(sampling, "distribution", eager_to_end)
    tokens, _timing, ended = core.perform(models, context_of(made, 20), None, 16, 6, 11, [], SETTINGS)
    assert ended and len(tokens) == 6


def test_a_song_sung_with_cfg_needs_its_negative_prompt(models, song):
    made, _sound = song
    with pytest.raises(ValueError, match="negative prompt"):
        core.perform(models, context_of(made, 20), None, 4, 4, 11, [], dict(SETTINGS, cfg_scale=2.0))


def test_the_join_scores_are_the_models_log_probabilities_of_the_old_frames(models, song):
    made, _sound = song
    context = context_of(made, 20)
    tokens = [value + CODEC_OFFSET for value in codec_of(16)]
    suffix = [value + CODEC_OFFSET for value in made.codec[32:40]]
    scores = core.joins(models, context, tokens, 8, 16, suffix)
    assert list(scores) == list(range(8, 17))
    for count, score in scores.items():
        sequence = context + tokens[:count] + suffix
        with torch.inference_mode():
            logits = models.lm(torch.tensor([sequence])).logits[0]
        at = len(context) + count - 1
        span = logits[at:at + len(suffix), CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE].float()
        target = torch.tensor([token - CODEC_OFFSET for token in suffix])
        direct = float(torch.log_softmax(span, -1).gather(1, target[:, None]).mean())
        assert score == pytest.approx(direct, abs=1e-5)


def test_the_codec_head_answers_with_the_same_rows_from_a_window_as_from_the_whole_head(lm):
    """Under 'auto' the head on the card is a window of the table from MUSIC_END, and the join must
    multiply by the codec rows alone either way."""
    from yue2_comfy import vocabulary

    plain = lm.lm_head
    hidden = torch.randn(3, plain.weight.shape[1])
    with core._codec_head(lm):
        whole = lm.lm_head(hidden)
    assert lm.lm_head is plain
    window = vocabulary.SlicedHead(plain, torch.device("cpu"))
    window.retune(MUSIC_END, CODEC_OFFSET + CODEC_SIZE)
    window.windowed = True
    lm.lm_head = window
    try:
        with core._codec_head(lm):
            sliced = lm.lm_head(hidden)
        assert lm.lm_head is window
    finally:
        lm.lm_head = plain
    assert tuple(whole.shape) == (3, CODEC_SIZE)
    assert torch.equal(sliced, whole)


def test_a_prefill_in_pieces_says_what_one_piece_says(models, song, monkeypatch):
    made, _sound = song
    context = context_of(made, 60)
    tokens = [value + CODEC_OFFSET for value in codec_of(16)]
    suffix = [value + CODEC_OFFSET for value in made.codec[72:80]]
    whole = core.joins(models, context, tokens, 8, 16, suffix)
    monkeypatch.setattr(core, "PREFILL_AREA", 17 * len(context))
    monkeypatch.setattr(core, "PIECE_FLOOR", 1)
    assert core.piece_for(len(context) + 8) < len(context)
    pieces = core.joins(models, context, tokens, 8, 16, suffix)
    assert list(pieces) == list(whole)
    for count in whole:
        assert pieces[count] == pytest.approx(whole[count], abs=1e-5)


def test_the_graph_prefill_in_pieces_fills_the_cache_one_piece_would(models, song):
    made, _sound = song
    context = context_of(made, 60)
    original = cuda_graph.GraphAR
    plain = cuda_graph.GraphAR(models.lm, [context], 4, capture=False)
    expected = plain.prefill()
    with core._in_pieces(len(context) + 1):
        short = cuda_graph.GraphAR(models.lm, [context], 4, capture=False)
        assert torch.equal(short.prefill(), expected)
    with core._in_pieces(16):
        pieces = cuda_graph.GraphAR(models.lm, [context], 4, capture=False)
        got = pieces.prefill()
    assert cuda_graph.GraphAR is original and pieces.ready
    assert torch.allclose(got, expected, atol=1e-5)
    used = len(context)
    for mine, theirs in zip(pieces.keys + pieces.values, plain.keys + plain.values):
        assert torch.allclose(mine[:, :used], theirs[:, :used], atol=1e-5)


def test_the_window_lies_just_outside_what_the_decoder_hears_of_the_edit():
    samples = 200 * HOP - 64
    first, end, left, right = core._window(200, (80, 100), HALO, HOP, samples)
    assert left == ((80 - HALO) * HOP - ops.FADE_SAMPLES, (80 - HALO) * HOP)
    assert right == ((100 + HALO) * HOP, (100 + HALO) * HOP + ops.FADE_SAMPLES)
    assert first * HOP <= left[0] and end * HOP >= right[1]
    first, end, left, right = core._window(200, (2, 30), HALO, HOP, samples)
    assert left is None and first == 0
    first, end, left, right = core._window(200, (180, 200), HALO, HOP, samples)
    assert right is None and end == 200


@pytest.mark.parametrize("start,stop,count", [(80, 100, 20), (80, 100, 27), (80, 100, 11), (80, 120, 0),
                                              (1, 20, 25), (170, 200, 30), (150, 200, 0)])
def test_the_new_sound_is_the_whole_edited_song_decoded(models, start, stop, count):
    old_latents = torch.randn((200, 64), generator=torch.Generator().manual_seed(4))
    new_part = torch.randn((count, 64), generator=torch.Generator().manual_seed(5))
    edited = torch.cat([old_latents[:start], new_part, old_latents[stop:]])
    low, high = max(0, start - ops.MARGIN), min(len(edited), start + count + ops.MARGIN)
    edited[low:high] += 0.5
    old = decode_locally(old_latents)[0]
    region = ops.Region(start, stop, count, 0)
    result, _timing = core.splice(models, old, edited, region, count)
    whole = decode_locally(edited)
    assert result.shape == whole.shape == (1, 2, old.shape[-1] - (stop - start - count) * HOP)
    frames = len(edited)
    _first, _end, left, right = core._window(frames, (low, high), HALO, HOP, whole.shape[-1])
    fades = torch.zeros(whole.shape[-1], dtype=torch.bool)
    for fade in (left, right):
        if fade is not None:
            fades[fade[0]:fade[1]] = True
    assert torch.equal(result[..., ~fades], whole[..., ~fades])
    assert torch.allclose(result[..., fades], whole[..., fades], atol=1e-6)


def test_retakes_sing_the_stretch_again_and_keep_the_rest_of_the_song(models, song):
    made, sound = song
    region = ops.retake(20, 32, made.frames, len(made.prefix))
    takes = core.retakes(models, made, sound, region, [1, 2], SETTINGS)
    old = core.latents_of(made)
    for seed, take in zip((1, 2), takes):
        count = take.count
        assert region.shortest <= count <= region.longest
        assert take.join == take.joins[count] == max(take.joins.values())
        assert list(take.song.codec[:20]) == list(made.codec[:20])
        assert list(take.song.codec[20 + count:]) == list(made.codec[32:])
        assert take.song.noise == ops.edited_noise(made.noise, region, seed, count)
        new = core.latents_of(take.song)
        assert torch.equal(new[:20 - ops.MARGIN], old[:20 - ops.MARGIN])
        assert torch.equal(new[20 + count + ops.MARGIN:], old[32 + ops.MARGIN:])
        assert take.waveform.shape[-1] == sound.shape[-1] - (12 - count) * HOP
        assert torch.equal(take.waveform[..., :HOP], sound[..., :HOP])
        assert torch.equal(take.waveform[..., -HOP:], sound[..., -HOP:])
        assert take.song.origin == core.ORIGIN
        assert (take.song.prefix, take.song.lyrics, take.song.score) == (made.prefix, made.lyrics, made.score)
    assert core.best(takes) == max(range(2), key=lambda index: takes[index].join)
    again = core.retakes(models, made, sound, region, [1], SETTINGS)[0]
    assert list(again.song.codec) == list(takes[0].song.codec)
    assert torch.equal(again.waveform, takes[0].waveform)


def test_a_retake_to_the_end_of_the_song_has_no_join_to_choose(models, song):
    made, sound = song
    region = ops.retake(90, FRAMES, made.frames, len(made.prefix))
    take = core.retakes(models, made, sound, region, [1], SETTINGS)[0]
    assert take.joins == {} and take.join is None and not take.ended
    assert take.count == region.length
    assert take.waveform.shape[-1] == sound.shape[-1]


def test_a_cut_joins_the_two_sides_under_the_new_words(models, song):
    made, sound = song
    region = ops.cut(40, 60, made.frames)
    take = core.cut(models, made, sound, region, "[verse]\nla", "X:2", SETTINGS)
    assert list(take.song.codec) == list(made.codec[:40]) + list(made.codec[60:])
    assert list(take.song.prefix) == prompt("[verse]\nla", Words().encode("X:2"))
    assert (take.song.lyrics, take.song.score, take.count, take.join) == ("[verse]\nla", "X:2", 0, None)
    assert take.song.noise == [[SEED, 0, 40], [SEED, 60, FRAMES - 60]]
    old, new = core.latents_of(made), core.latents_of(take.song)
    assert torch.equal(new[:40 - ops.MARGIN], old[:40 - ops.MARGIN])
    assert torch.equal(new[40 + ops.MARGIN:], old[60 + ops.MARGIN:])
    assert take.waveform.shape[-1] == sound.shape[-1] - 20 * HOP


def test_a_song_whose_sound_is_not_its_latents_is_not_edited(models, song):
    made, sound = song
    region = ops.retake(20, 32, made.frames, len(made.prefix))
    alone = songs.made("YuE2 Generate Song", "s", "l", SEED, dict(SETTINGS, vocals_only=True), "X:1",
                       list(made.prefix), None, list(made.codec), made.noise, made.latents, 48000, 2,
                       sound.shape[-1])
    with pytest.raises(ValueError, match="vocals_only"):
        core.retakes(models, alone, sound, region, [1], SETTINGS)
    with pytest.raises(ValueError, match="channels by"):
        core.retakes(models, made, sound[..., :-1], region, [1], SETTINGS)
    with pytest.raises(ValueError, match="not a stretch"):
        core.cut(models, made, sound, ops.Region(100, FRAMES + 1, 0, 0), "l", "X:1", SETTINGS)


@pytest.mark.parametrize("pieces", [
    [(0, 20), (80, FRAMES), (20, 80)],
    [(0, 20), (60, 65), (20, 60), (65, FRAMES)],
    [(90, FRAMES), (0, 90)],
])
def test_a_move_keeps_every_frame_and_draws_only_its_seams_again(models, song, pieces):
    """The old sound in its new order, with the seams decoded: the whole moved song decoded.

    The first puts the song's last frames, whose sound is 64 samples short, in the middle; the
    second has two seams closer than a window, which share one.
    """
    made, sound = song
    moved = sum(stop - start for start, stop in pieces[1:2])
    take = core.moved(models, made, sound, pieces, moved, "[verse]\nla", "X:2", SETTINGS)
    assert list(take.song.codec) == [value for start, stop in pieces for value in made.codec[start:stop]]
    assert take.song.noise == ops.moved_noise(made.noise, pieces)
    assert list(take.song.prefix) == prompt("[verse]\nla", Words().encode("X:2"))
    assert (take.seed, take.count, take.join, take.ended) == (0, moved, None, False)
    old, new = core.latents_of(made), core.latents_of(take.song)
    known = torch.cat([old[start:stop] for start, stop in pieces])
    held = torch.ones(FRAMES, dtype=torch.bool)
    for seam in core.seams_of(pieces):
        held[max(0, seam - ops.MARGIN):seam + ops.MARGIN] = False
    assert torch.equal(new[held], known[held])
    assert not torch.equal(new[~held], known[~held])
    assert take.waveform.shape == sound.shape
    assert torch.allclose(take.waveform, decode_locally(new), atol=1e-6)


def test_the_seams_of_a_move_are_where_pieces_meet_that_were_not_neighbours():
    assert core.seams_of([(0, 20), (80, 120), (20, 80)]) == [20, 60]
    assert core.seams_of([(0, 20), (20, 80)]) == []
    assert core.seams_of([(90, 120), (0, 90)]) == [30]


def test_a_move_must_hold_every_frame_of_the_song(models, song):
    made, sound = song
    with pytest.raises(ValueError, match="hold 100 frames and the song has 120"):
        core.moved(models, made, sound, [(0, 20), (40, 120)], 80, "l", "X:1", SETTINGS)


MOVED = [(0, 40), (80, FRAMES), (40, 80)]
"""The last 40 frames of the tiny song put after its first 40: seams at frames 40 and 80."""


def test_a_move_sings_the_stretch_before_each_seam_again_and_keeps_the_rest(models, song):
    """Each on the song the one before left; the audio of the whole song drawn once, the sound laid in."""
    made, sound = song
    take = core.moved(models, made, sound, MOVED, 40, "[verse]\nla", "X:2", SETTINGS,
                      sung=((25, 40), (65, 80)))
    spliced = [value for start, stop in MOVED for value in made.codec[start:stop]]
    (low, high, first), (again, end, second) = take.sung
    region = ops.retake(25, 40, FRAMES, len(take.song.prefix))
    assert (low, high) == (25, 40) and region.shortest <= first <= region.longest
    assert (again, end) == (65 + first - 15, 80 + first - 15)
    codec = list(take.song.codec)
    assert codec[:25] == spliced[:25]
    assert codec[25 + first:again] == spliced[40:65]
    assert codec[again + second:] == spliced[80:]
    frames = FRAMES + first - 15 + second - 15
    assert take.song.frames == frames and sum(run[2] for run in take.song.noise) == frames
    assert take.waveform.shape[-1] == sound.shape[-1] + (frames - FRAMES) * HOP
    new = core.latents_of(take.song)
    known = torch.cat([core.latents_of(made)[start:stop] for start, stop in MOVED])
    assert torch.equal(new[:25 - ops.MARGIN], known[:25 - ops.MARGIN])
    assert torch.equal(new[again + second + ops.MARGIN:], known[80 + ops.MARGIN:])
    assert torch.allclose(take.waveform, decode_locally(new), atol=1e-6)
    assert (take.seed, take.count, take.join) == (0, 40, None)
    twice = core.moved(models, made, sound, MOVED, 40, "[verse]\nla", "X:2", SETTINGS,
                       sung=((25, 40), (65, 80)))
    assert list(twice.song.codec) == codec and torch.equal(twice.waveform, take.waveform)


def test_the_stretch_sung_again_comes_out_as_long_as_puts_the_beat_back(models, song, monkeypatch):
    """Where the beat is read across the seam, only the lengths that put it back are open to the join."""
    from yue2_comfy.constants import FRAME_SECONDS
    from yue2_comfy.inpaint import beat

    made, sound = song
    read = []

    def jump(sound_read, rate, second, pulse):
        read.append((sound_read.shape[-1], second, pulse))
        return 0.09

    monkeypatch.setattr(beat, "jump", jump)
    take = core.moved(models, made, sound, MOVED, 40, "[verse]\nla", "X:2", SETTINGS,
                      sung=((25, 40),), pulse=0.25)
    assert read == [(FRAMES * HOP, 40 * FRAME_SECONDS, 0.25)], "the splice's own sound, at the seam"
    (_low, _high, count), = take.sung
    region = ops.retake(25, 40, FRAMES, len(take.song.prefix))
    allowed = beat.counts_on_beat(15, region.shortest, region.longest, 0.09, 0.25, FRAME_SECONDS)
    assert allowed == [13, 19] and count in allowed



HEAD = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:C\n% verse\nV: Vocal\nC16|D16|\n"
        "V: Ins\nZ|Z|\n% interlude\n")
"""A score the model writes a break on from: two bars of a verse and the section of playing begun."""

BUILT = HEAD + "V: Vocal\nz16|z16|\nV: Ins\nC16|E16|\n"
"""The score with the break laid in: the verse and two bars of playing."""


def a_writer(written, failing=0):
    """A stand-in for the model writing a break: what it was asked, and the head back with a mark."""
    def score_on(models, song, lyrics, head, seed, settings, progress=None, band=None,
                 cancelled=None, most=None):
        written.append((seed, most, lyrics))
        return head + ("bad" if len(written) <= failing else "V: Vocal\nz16|\nV: Ins\nC16|\n")
    return score_on


def test_a_break_plays_new_frames_between_two_old_ones_and_keeps_the_rest(models, song, monkeypatch):
    made, sound = song
    written, built = [], []
    monkeypatch.setattr(core, "score_on", a_writer(written))

    def build(text):
        built.append(text)
        return BUILT

    takes = core.broke(models, made, sound, 40, HEAD, "[verse]\nla\n\n[Interlude]", 2, [1, 2],
                       SETTINGS, build, lambda sheet: 12)
    room = math.ceil(len(Words().encode(HEAD)) / 2.0 * 2 * core.BREAK_TOKENS) + 32
    assert [(seed, most) for seed, most, _lyrics in written] == [(1, room), (2, room)], (
        "each take writes the break's bars from its own seed, with room for a few bars only")
    assert built == [HEAD + "V: Vocal\nz16|\nV: Ins\nC16|\n"] * 2, "what the model wrote is laid in"
    old = core.latents_of(made)
    region = ops.insert(40, FRAMES, 12, len(takes[0].song.prefix))
    for seed, take in zip((1, 2), takes):
        count = take.count
        assert region.shortest <= count <= region.longest
        assert take.join == take.joins[count] == max(take.joins.values()), "the join chose freely"
        assert list(take.song.codec[:40]) == list(made.codec[:40])
        assert list(take.song.codec[40 + count:]) == list(made.codec[40:]), "nothing old is gone"
        assert take.song.noise == ops.edited_noise(made.noise, region, seed, count)
        new = core.latents_of(take.song)
        assert torch.equal(new[:40 - ops.MARGIN], old[:40 - ops.MARGIN])
        assert torch.equal(new[40 + count + ops.MARGIN:], old[40 + ops.MARGIN:])
        assert take.waveform.shape[-1] == sound.shape[-1] + count * HOP
        assert torch.equal(take.waveform[..., :HOP], sound[..., :HOP])
        assert torch.equal(take.waveform[..., -HOP:], sound[..., -HOP:])
        assert (take.song.lyrics, take.song.score) == ("[verse]\nla\n\n[Interlude]", BUILT)
        assert take.voice is None and not take.ended
    assert list(takes[0].song.prefix) == prompt("[verse]\nla\n\n[Interlude]",
                                                Words().encode(takes[0].song.score))


def test_a_break_that_cannot_be_laid_into_the_score_is_written_again_with_more_room(models, song,
                                                                                    monkeypatch):
    made, sound = song
    written = []
    monkeypatch.setattr(core, "score_on", a_writer(written, failing=1))

    def build(text):
        if text.endswith("bad"):
            raise ValueError("the model wrote 0 whole bars of the break where 2 were asked for")
        return BUILT

    take = core.broke(models, made, sound, 40, HEAD, "[verse]\nla", 2, [5], SETTINGS, build,
                      lambda sheet: 12)[0]
    assert [seed for seed, _most, _lyrics in written] == [5, 5 + core.SCORE_SEED_STEP]
    assert written[1][1] == 2 * written[0][1]
    assert take.seed == 5 and take.song.score == BUILT, "the take keeps its own seed"
    monkeypatch.setattr(core, "score_on", a_writer([], failing=core.SCORE_TRIES))
    with pytest.raises(ValueError, match="none of them could be laid"):
        core.broke(models, made, sound, 40, HEAD, "[verse]\nla", 2, [5], SETTINGS, build,
                   lambda sheet: 12)


def test_a_take_with_no_voice_left_in_its_break_is_kept_before_one_that_sings():
    def take(voice, join):
        return core.Take(seed=0, waveform=None, song=None, count=1, join=join, joins={},
                         ended=False, timing={}, voice=voice)

    assert core.best([take(-5.0, -3.0), take(-30.0, -5.0)]) == 1, "a voice outweighs any join"
    assert core.best([take(-30.0, -5.0), take(-60.0, -4.0)]) == 1, "among clean ones, the join"
    assert core.best([take(-2.0, -3.0), take(-9.0, -5.0)]) == 1, "among singing ones, the quieter"
    assert core.best([take(None, -3.0), take(None, -5.0)]) == 0, "unheard, the join"
    assert core.voiced(take(core.VOICE_LEFT + 0.1, 0)) and not core.voiced(take(core.VOICE_LEFT, 0))
