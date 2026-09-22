import xml.etree.ElementTree as ET

import pytest

from eduai.data import openstax as ox

NS = 'xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML"'


def _el(xml: str) -> ET.Element:
    return ET.fromstring(xml.replace("<ROOT", f"<ROOT {NS}"))


def test_inline_text_converts_math_and_drops_parenthesized_figure_refs():
    el = _el(
        "<ROOT>Speed is <m:math><m:msub><m:mi>v</m:mi><m:mn>0</m:mn></m:msub></m:math> at first "
        '(<link target-id="fig1"/>). Water is H<sub>2</sub>O.</ROOT>'
    )
    assert ox.inline_text(el) == "Speed is v_0 at first. Water is H2O."


def test_inline_text_rejects_a_reference_inside_the_sentence_and_figures():
    with pytest.raises(ox.Unrenderable):
        ox.inline_text(_el('<ROOT><link target-id="fig1"/> shows a grindstone.</ROOT>'))
    with pytest.raises(ox.Unrenderable):
        ox.inline_text(_el("<ROOT>See <figure/> here.</ROOT>"))


def test_parse_mcq_list_and_lettered_paragraph_formats():
    listed = _el(
        "<ROOT><problem><para>Atoms that vary in neutrons are called ____.</para>"
        '<list number-style="lower-alpha"><item>ions</item><item>neutrons</item>'
        "<item>neutral atoms</item><item>isotopes</item></list></problem>"
        "<solution><para>D</para></solution></ROOT>"
    )
    q = ox.parse_mcq(listed)
    assert q["key"] == "D" and q["choices"]["D"] == "isotopes" and q["stem"].startswith("Atoms")
    lettered = _el(
        "<ROOT><problem><para>Substance A is shiny. It is likely a(n):</para>"
        "<para>(a) ionic solid</para><para>(b) metallic solid</para>"
        "<para>(c) molecular solid</para><para>(d) covalent network solid</para></problem>"
        "<solution><para>(b) metallic solid</para></solution></ROOT>"
    )
    assert ox.parse_mcq(lettered)["key"] == "B"
    # five options, or a solution naming two letters, is not a usable four-option reference
    five = lettered.find("{http://cnx.rice.edu/cnxml}problem")
    extra = ET.SubElement(five, "{http://cnx.rice.edu/cnxml}para")
    extra.text = "(e) none"
    assert ox.parse_mcq(lettered) is None


def test_chunks_merge_short_paragraphs_but_not_across_gaps_or_backrefs():
    s = "A short sentence that is about forty chars."
    paras = [
        ("s1", s),
        ("s1", s),
        ("s1", s),
        ("s1", s),
        ("s1", None),
        ("s1", "This relation is Ohm's law. " + s * 4),
    ]
    out = ox.chunks(paras)
    assert out == [" ".join([s] * 4)]


def test_fit_length_keeps_leading_sentences_and_the_minimum():
    s = "Sentence number one is here and fairly long. " * 8
    fitted = ox.fit_length(s.strip(), 200)
    assert ox.MIN_CHARS <= len(fitted) <= 250 and fitted.endswith(".")


def test_prose_filter():
    assert ox.prose("Energy is conserved.")
    assert not ox.prose("where v is the speed.")
    assert not ox.prose("Under that condition, the equation becomes")


def test_parse_mcq_keeps_a_roman_numeral_statement_list_in_the_stem():
    roman = _el(
        "<ROOT><problem><para>Which of the following raise the resonant frequency?</para>"
        '<list number-style="upper-roman"><item>Add water.</item><item>Use a denser fluid.</item>'
        "<item>Warm the room.</item></list>"
        '<list number-style="lower-alpha"><item>I only</item><item>I and III</item>'
        "<item>II and III</item><item>all of the above</item></list></problem>"
        "<solution><para>(b)</para></solution></ROOT>"
    )
    q = ox.parse_mcq(roman)
    assert q["key"] == "B" and q["choices"]["B"] == "I and III"
    assert q["stem"].endswith("frequency? I. Add water. II. Use a denser fluid. III. Warm the room.")
    other = roman.find("{http://cnx.rice.edu/cnxml}problem").find("{http://cnx.rice.edu/cnxml}list")
    other.set("number-style", "arabic")
    assert ox.parse_mcq(roman) is None
