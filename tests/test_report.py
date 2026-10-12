from exoplanets.pipeline import run
from tests.synthetic import make_planets, write_archive_style_csv


def test_pipeline_writes_a_self_contained_report(tmp_path):
    csv = write_archive_style_csv(make_planets(n_systems=900, seed=3), tmp_path / "synthetic.csv")
    out = run(csv, tmp_path / "report.html", synthetic=True, n_boot=50)
    html = out.read_text(encoding="utf-8")

    assert "SYNTHETIC TEST DATA" in html
    assert "<picture>" in html and html.count("data:image/png;base64,") >= 10
    assert 'src="http' not in html and "src='http" not in html
    assert "—" not in html, "report text should not contain em dashes"
    for heading in ("How discoveries accumulated", "gap in the sizes", "Data provenance"):
        assert heading in html
    # Table views must be real tables, not escaped markup shown as text.
    assert html.count("<table>") == 3
    assert "&lt;table" not in html and "&lt;picture" not in html


def test_real_data_report_has_no_synthetic_banner(tmp_path):
    csv = write_archive_style_csv(make_planets(n_systems=600, seed=4), tmp_path / "data.csv")
    out = run(csv, tmp_path / "report.html", synthetic=False, n_boot=30)
    assert "SYNTHETIC TEST DATA" not in out.read_text(encoding="utf-8")
