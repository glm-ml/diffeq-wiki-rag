from app.ingest.chunking import chunk_sections, save_corpus, load_corpus
from app.ingest.wikipedia import ArticleSpec, extract_sections

ARTICLE_HTML = """
<div class="mw-parser-output">
  <p>Введение в статью о непрерывности Липшица.</p>
  <table class="sidebar"><tr><td>NOISE_NAVIGATION</td></tr></table>
  <div class="mw-heading mw-heading2"><h2 id="Definition">Definition</h2></div>
  <p>
    Функция удовлетворяет условию
    <span class="mwe-math-element">
      <span class="mwe-math-mathml-a11y" style="display:none">L</span>
      <img class="mwe-math-fallback-image-inline"
           alt="{\\displaystyle |f(x)-f(y)|\\leq L|x-y|}" />
    </span>
    для всех x и y.
  </p>
  <p>См. <a href="/wiki/Metric_space">metric spaces</a> и <a href="/wiki/Lipschitz_continuity">Lipschitz continuity</a>.</p>
  <img class="mwe-math-fallback-image-display"
       alt="{\\displaystyle {\\begin{aligned}a&=b\\\\c&=d\\end{aligned}}}" />
  <div class="mw-heading mw-heading3"><h3 id="Local_Lipschitz">Local Lipschitz</h3></div>
  <p>Локальное условие Липшица выполняется на ограниченных множествах.</p>
  <div class="mw-heading mw-heading2"><h2 id="References">References</h2></div>
  <ol class="references"><li>Источник, который не должен попасть в корпус.</li></ol>
</div>
"""


def test_extract_sections_keeps_content_and_skips_service_sections():
    sections = extract_sections(ARTICLE_HTML, ArticleSpec("Lipschitz continuity"))

    titles = [section.subsection or section.section for section in sections]
    assert "Definition" in titles
    assert "Local Lipschitz" in titles
    assert all("References" not in title for title in titles)

    definition = next(section for section in sections if section.section == "Definition")
    assert "#Definition" in definition.url
    assert "|f(x)-f(y)|\\leq L|x-y|" in definition.text
    assert "{\\displaystyle" not in definition.text
    assert "\\begin{aligned}" in definition.text
    assert "{\\begin{aligned}" not in definition.text
    assert "См. metric spaces и Lipschitz continuity." in definition.text

    all_text = " ".join(section.text for section in sections)
    assert "NOISE_NAVIGATION" not in all_text
    assert "Источник, который не должен попасть" not in all_text


def test_chunking_splits_long_sections_and_persists(tmp_path):
    from app.ingest.wikipedia import WikiSection

    section = WikiSection(
        article="Lipschitz continuity",
        section="Definition",
        subsection="",
        url="https://en.wikipedia.org/wiki/Lipschitz_continuity#Definition",
        text="Функция удовлетворяет условию Липшица. " * 40,
    )
    chunks = chunk_sections([section], chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["article"] == "Lipschitz continuity" for chunk in chunks)
    assert all(chunk.metadata["source"].startswith("https://en.wikipedia.org/") for chunk in chunks)
    assert chunks[0].metadata["chunk_id"].startswith("Lipschitz continuity::Definition::")

    path = tmp_path / "corpus.jsonl"
    save_corpus(chunks, path)
    restored = load_corpus(path)

    assert len(restored) == len(chunks)
    assert restored[0].page_content == chunks[0].page_content
    assert restored[0].metadata["section"] == "Definition"
