"""Lyrics laid along a bare tune: syllables, phrases, which bars each line is sung on, and what the node says."""

from __future__ import annotations

import re
from fractions import Fraction

from yue2_comfy import phrasing
from yue2_comfy.sheetsage import abc_rebuild
from yue2_comfy.vendor.yue2_music import abc_tools

LA = "la la la la la la la"
NA = "na na na na na na na"
LO = "lo lo lo lo lo lo lo"
PHRASE = "z2C2D2E2F2G2A2B2|"
ANSWER = "z2c2B2A2G2F2E2D2|"
RIFF = "z4c4G8|"


def bars_in(line: str) -> int:
    count = 0
    for bar in line[:-1].split("|"):
        found = re.fullmatch(r"Z([2-4])?", bar.strip())
        count += int(found.group(1) or "1") if found else 1
    return count


def tune(*lines, meter="4/4", unit=16, bpm=100, key="C") -> str:
    """A score in YuE2's dialect, the instrument silent unless a line says otherwise.

    Each line is the vocal bars as text, or a dict with ``vocal`` and optional
    ``ins``, ``fields`` (``M:`` and ``K:`` lines) and ``comments``.
    """
    text = ["X:1", "T:", "M:" + meter, "L:1/{}".format(unit), "Q:1/4={}".format(bpm),
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
            'V: Ins clef=treble name="Ins Melody" snm="Inst."', "K:" + key]
    for entry in lines:
        entry = entry if isinstance(entry, dict) else {"vocal": entry}
        count = bars_in(entry["vocal"])
        text.extend("% " + label for label in entry.get("comments", ()))
        for voice, music in (("Vocal", entry["vocal"]), ("Ins", entry.get("ins", "Z{}|".format(count) if count > 1
                                                                          else "Z|"))):
            text.append("V: " + voice)
            text.extend(entry.get("fields", ()))
            text.append(music)
    return "\n".join(text) + "\n"


def sung(score: str) -> list:
    return [pitch for _onset, pitch, _length in abc_tools.parse(score).voices["Vocal"].notes]


def comments(score: str) -> list:
    return [line[2:] for line in score.splitlines() if line.startswith("% ")]


def test_syllables_are_counted_from_the_vowels_of_each_script():
    assert phrasing.syllables("Hold the tune and let it ring") == 7
    assert phrasing.syllables("played wanted fades boxes table whole") == 9
    assert phrasing.syllables("\u041f\u0440\u0438\u0432\u0435\u0442, \u043a\u0430\u043a \u0434\u0435\u043b\u0430") == 5
    assert phrasing.syllables("\u4f60\u597d\u4e16\u754c") == 4
    assert phrasing.syllables("caf\u00e9 na\u00efve") == 4
    assert phrasing.syllables("-- 42 --") == 0


def test_the_words_songs_say_differently_from_their_spelling_are_counted_as_sung():
    """'ev-ry' and 'go-ing': a line one syllable off takes a phrase one note off, and a spare note gets a made-up word."""
    counts = {"every": 2, "everything": 3, "everyone": 3, "different": 2, "going": 2, "trying": 2, "sing": 1,
              "piano": 3, "radio": 3, "million": 2, "nation": 2, "special": 2, "serious": 3, "people": 2}
    assert {word: phrasing.syllables(word) for word in counts} == counts
    assert phrasing.syllables("Every street remembers night") == 7


def test_tags_become_the_labels_of_the_models_scores():
    assert phrasing.label_of("Verse 2") == "verse"
    assert phrasing.label_of("Pre-Chorus") == "pre-chorus"
    assert phrasing.label_of("prechorus") == "pre-chorus"
    assert phrasing.label_of("Hook") == "chorus"
    assert phrasing.label_of("Outro:") == "outro"
    assert phrasing.label_of("Something else") == "verse"


def test_lyrics_are_sections_of_sung_lines_and_lines_before_a_tag_are_a_verse():
    found = phrasing.lyric_sections("first line\n\n[Intro]\n\n[Chorus 2]\nsing this\n...\nand this\n")
    assert [(section["label"], section["lines"]) for section in found] == [
        ("verse", ["first line"]), ("intro", []), ("chorus", ["sing this", "and this"])]
    assert found[2]["tag"] == "Chorus 2"


def test_a_line_ends_at_the_bar_line_where_the_tune_runs_on_into_a_riff():
    """The GTA intro's last sung phrase runs into the riff without a breath; the line keeps to its own bar."""
    score = tune("z2e2g2e2d3d2cd2|z2B2B2B2c2G2c2G2|d4G8G4|z4G,8z4|")
    laid = phrasing.lay(score, "[Verse]\nNeon fades along the lane\nEvery street remembers night")
    assert sung(laid.score) == sung(tune("z2e2g2e2d3d2cd2|z2B2B2B2c2G2c2G2|"))
    assert "[Verse] on bars 1-2" in laid.notices[0][1]


def test_a_breath_or_a_held_note_ends_a_phrase_and_a_long_phrase_is_cut_at_its_bars():
    notes = [[Fraction(0), 60, Fraction(1, 2)], [Fraction(1, 2), 62, Fraction(1, 4)],
             [Fraction(1), 64, Fraction(1, 2)],
             [Fraction(2), 65, Fraction(2)], [Fraction(4), 67, Fraction(1)]]
    found = phrasing.pieces(notes, [Fraction(0), Fraction(4)])
    assert [(piece.start, piece.end, piece.notes, piece.breath) for piece in found] == [
        (Fraction(0), Fraction(3, 2), 3, True), (Fraction(2), Fraction(4), 1, True), (Fraction(4), Fraction(5), 1, True)]
    long_line = [[Fraction(beat, 2), 60, Fraction(1, 2)] for beat in range(24)]
    cut = phrasing.pieces(long_line, [Fraction(0), Fraction(4), Fraction(8)])
    assert [(piece.notes, piece.breath) for piece in cut] == [(8, True), (8, False), (8, False)]


def test_a_score_that_names_a_section_or_cannot_be_read_is_sung_as_it_came():
    words = "[Verse]\n" + LA
    assert phrasing.lay(tune({"vocal": PHRASE, "comments": ["verse"]}), words) is None
    assert phrasing.lay("X:1\nK:C\nCDEF|", words) is None
    assert phrasing.lay(tune("Z|"), words) is None
    assert phrasing.lay(tune(PHRASE), "[Intro]\n\n[Instrumental]") is None


def test_each_line_takes_a_phrase_that_fits_a_riff_is_passed_over_and_the_tune_comes_round_again():
    """The GTA intro in small: four silent bars, four sung phrases of seven notes, then a riff of two notes a bar."""
    score = tune("Z4|", PHRASE + ANSWER + PHRASE + ANSWER, RIFF * 4)
    laid = phrasing.lay(score, "[Verse]\n" + "\n".join([LA] * 4) + "\n\n[Chorus]\n" + "\n".join([NA] * 4))
    parsed = abc_tools.parse(laid.score)
    verse = sung(tune(PHRASE + ANSWER + PHRASE + ANSWER))
    assert comments(laid.score) == ["intro", "verse", "chorus"]
    assert sung(laid.score) == verse + verse
    assert len(parsed.voices["Vocal"].bars) == 4 + 4 + 4 + 1
    last = parsed.voices["Vocal"].notes[-1]
    assert last[0] + last[2] == parsed.voices["Vocal"].time - 4, "one empty bar closes the song"
    assert laid.seconds == 13 * 2.4
    assert laid.notices == [("notice", phrasing.LAID.format(
        where="bars 1-4 as the intro; [Verse] on bars 5-8; [Chorus] on bars 5-8",
        unsung=phrasing.UNSUNG.format(bars="bars 9-12"), seconds=laid.seconds, ceiling=phrasing.ceiling(laid.seconds)))]
    assert "31 seconds in; with 'max_seconds' at 0 the singing is stopped at 36 seconds" in laid.notices[0][1]


def test_a_chorus_sung_again_is_sung_on_its_own_bars_again():
    score = tune("z2C2D2E2F2G2A2B2|z2D2E2F2G2A2B2c2|z2E2F2G2A2B2c2d2|z2F2G2A2B2c2d2e2|")
    laid = phrasing.lay(score, "[Verse]\n{}\n[Chorus]\n{}\n[Verse]\n{}\n[Chorus]\n{}".format(LA, NA, LO, NA))
    bars = [sung(tune(bar + "|")) for bar in score.splitlines()[9].split("|")[:-1]]
    assert sung(laid.score) == bars[0] + bars[1] + bars[2] + bars[1]
    assert comments(laid.score) == ["verse", "chorus", "verse", "chorus"]
    assert "[Verse] on bar 1; [Chorus] on bar 2; [Verse] on bar 3; [Chorus] on bar 2" in laid.notices[0][1]


def test_notes_of_a_phrase_nobody_sings_are_rested_in_a_bar_that_is_sung():
    """The third line goes back to the first phrase, so the one note after the second is never reached."""
    laid = phrasing.lay(tune("C2D2E2F2z8|G2A2B2c2z4d4|"), "[Verse]\nla la la la\nla la la la\nla la la la")
    assert sung(laid.score) == [60, 62, 64, 65, 67, 69, 71, 72, 60, 62, 64, 65]
    assert len(abc_tools.parse(laid.score).voices["Vocal"].bars) == 4


def test_chords_keys_and_meters_go_with_their_bars():
    score = tune({"vocal": '"C"z2C2D2E2F2G2A2B2|"G"z2B2A2G2F2E2D2C2|'},
                 {"vocal": '"D"z2D2E2F2G2A2|', "fields": ["M:3/4", "K:G"]})
    laid = phrasing.lay(score, "[Verse]\n{}\n{}\nla la la la la".format(LA, LA))
    vocal = abc_tools.parse(laid.score).voices["Vocal"]
    assert [symbol for _time, symbol in vocal.chords] == ["C", "G", "D"]
    assert [bar[2] for bar in vocal.bars] == [(4, 4), (4, 4), (3, 4), (3, 4)]
    assert [key for _time, key in vocal.keys] == ["C", "G"]
    assert sung(laid.score) == sung(score)


def test_a_tune_with_far_more_notes_than_the_words_have_syllables_is_warned_about():
    score = tune("CDEFGABcdcBAGFz2|CDEFGABcdcBAGFz2|", bpm=60)
    laid = phrasing.lay(score, "[Verse]\nla la la la\nla la la la")
    assert laid.notices[-1] == ("warn", phrasing.CROWDED_NOTES.format(ratio="3.5", notes=14, syllables=4))


def test_words_with_far_more_syllables_than_the_tune_has_notes_are_warned_about():
    laid = phrasing.lay(tune("z4C4z4D4|z4E4z4F4|z4G4z4A4|z4B4z4c4|"), "[Verse]\n" + "la " * 30)
    level, message = laid.notices[-1]
    assert level == "warn" and message == phrasing.CROWDED_WORDS.format(notes=1, syllables=30)


def test_a_words_line_that_fits_nothing_still_lands_and_the_score_still_reads():
    laid = phrasing.lay(tune(RIFF * 4), "[Verse]\n" + LA)
    assert abc_tools.parse(laid.score).voices["Vocal"].notes


def test_the_load_node_helpers_take_the_sections_out_or_name_one():
    score = tune({"vocal": PHRASE, "comments": ["chorus"]})
    assert phrasing.named(score) and not phrasing.named(phrasing.bare(score))
    assert phrasing.bare(score) == tune(PHRASE)
    assert phrasing.labelled(tune(PHRASE)) == tune({"vocal": PHRASE, "comments": ["verse"]})
    assert phrasing.labelled(score) == score
    assert phrasing.labelled("X:1\nnot a score") == "X:1\nnot a score"


def test_chord_symbols_go_back_to_the_labels_the_score_writer_takes():
    for symbol, label in (("C", "C:maj"), ("Bbm7/F", "Bb:min7/F"), ("F#m7b5", "F#:hdim7"), ("Ab6", "Ab:maj6"),
                          ("G7sus4", "G:sus4(b7)"), ("Dm(maj7)", "D:minmaj7")):
        assert phrasing.chord_label(symbol) == label
        assert abc_rebuild.chord_text(label) == symbol
