from crawler.crawl_en_source2 import parse_chapter_html


def test_parse_individual_verses_and_continuation():
    html = """
    <html><body><main>
      <h1>Second Chapter</h1>
      <p>The Blessed Lord said:</p>
      <p>1. First sentence of verse one</p>
      <p>continued sentence of verse one.</p>
      <p>2. Second verse text.</p>
      <p>Footnotes</p>
      <p>1. This is a footnote and must not be parsed.</p>
    </main></body></html>
    """
    verses, grouped = parse_chapter_html(html, 2, "https://example.org/chapter-2")
    assert [row["id"] for row in verses] == ["2.1", "2.2"]
    assert verses[0]["translations"]["en_source2"] == "First sentence of verse one continued sentence of verse one."
    assert grouped == []


def test_grouped_passage_is_not_duplicated_as_individual_verses():
    html = """
    <html><body><main>
      <h1>Second Chapter</h1>
      <p>42-44. This passage covers three verses together.</p>
      <p>45. This is a separate verse.</p>
      <p>The end of second chapter.</p>
    </main></body></html>
    """
    verses, grouped = parse_chapter_html(html, 2)
    assert [row["id"] for row in verses] == ["2.45"]
    assert len(grouped) == 1
    assert grouped[0]["start_verse"] == 42
    assert grouped[0]["end_verse"] == 44
    assert "three verses together" in grouped[0]["text"]


def test_invalid_chapter_is_rejected():
    try:
        parse_chapter_html("<p>1. Text</p>", 19)
    except ValueError as error:
        assert "between 1 and 18" in str(error)
    else:
        raise AssertionError("Expected invalid chapter to raise ValueError")
